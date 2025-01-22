from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

import streamlit as st
import os
from langchain_chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.chains.query_constructor.base import get_query_constructor_prompt
from langchain.chains.query_constructor.base import StructuredQueryOutputParser
from langchain_core.output_parsers import StrOutputParser

os.environ['OPENAI_API_KEY'] = ''


st.set_page_config(page_title="Search Your IMDB database.")
st.header="ASK ANYTHING ABOUT YOUR DB"
query=st.text_input("ask question here")

system_prompt = """
You are a helpful assistant specialized in answering questions.
Please provide answers only based on context below. If do not know the answer say I don't know.
Provide concise and accurate answers in natural language. Use the context below to answer the user's question:
{context}
"""


# Create a ChatPromptTemplate to include the system message
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{question}")
])


# Define the SQL agent
llm = ChatOpenAI(model="gpt-4o-mini")

metadata_field_info = [
    AttributeInfo(
        name="Title",
        description="This the Title of the movie",
        type="string",
    ),
    AttributeInfo(
        name="Certificate",
        description="This is the Certificate of the movie",
        type="string",
    ),
    AttributeInfo(
        name="Runtime",
        description="this is total Total length of the movie in mins",
        type="string",
    ),
    AttributeInfo(
        name="Genre1",
        description="This is the genre of the movie. This column has multiple values seperated by comma",
        type="string",
    ),
    AttributeInfo(
        name="Genre2",
        description="This is the genre of the movie. This column has multiple values seperated by comma",
        type="string",
    ),
    AttributeInfo(
        name="Genre3",
        description="This is the genre of the movie. This column has multiple values seperated by comma",
        type="string",
    ),
    AttributeInfo(
        name="Year",
        description="This the year when the movie was released.",
        type="integer",
    ),
    AttributeInfo(
        name="Director",
        description="This is the name of the directory of the movie",
        type="string",
    ),
    AttributeInfo(
        name="Rating", description="This is the rating from 1-10 received by the movie", type="float"
    ),
    AttributeInfo(
        name="MetaScore", description="this is Meta Score in range 1-100 received by the movie", type="integer"
    ),
    AttributeInfo(
        name="Votes", description="This is Number of Votes the movie received.", type="integer"
    ),
    AttributeInfo(
        name="Gross", description="This is the Revenue made by the movie.", type="integer"
    ),
    AttributeInfo(
        name="Star1", description="This is name of one of Actor's in the Movie", type="string"
    ),
    AttributeInfo(
        name="Star2", description="This is name of one of Actor's in the Movie", type="string"
    ),
    AttributeInfo(
        name="Star3", description="This is name of one of Actor's in the Movie", type="string"
    ),
    AttributeInfo(
        name="Star4", description="This is name of one of Actor's in the Movie", type="string"
    ),
]

document_content_description = "Brief summary of a movie"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectordb = Chroma(persist_directory="./chromdb", embedding_function=embeddings)

allowed_comparators = [
    "$eq",  # Equal to (number, string, boolean)
    "$ne",  # Not equal to (number, string, boolean)
    "$gt",  # Greater than (number)
    "$gte",  # Greater than or equal to (number)
    "$lt",  # Less than (number)
    "$lte",  # Less than or equal to (number)
    "$in",  # In array (string or number)
    "$nin",  # Not in array (string or number)
    "$exists", # Has the specified metadata field (boolean)
    "$and", # Combines multiple filters
]
# Examples for few-shot learning
examples = [
    (
        "What are the comedy movies released in 2019?",
        {
            "query": "",
            "filter": 'and(eq("Year", 2019), or(eq("Genre1", "Comedy"),eq("Genre2", "Comedy"),eq("Genre3", "Comedy")))',
        },
    ),
    (
        "What are the movies by actor Brad Pitt?",
        {
            "query": "",
            "filter": 'or(eq("Star1", "Brad Pitt"), eq("Star2", "Brad Pitt"),eq("Star3", "Brad Pitt"),eq("Star4", "Brad Pitt"))',
        },
    ),
    (
        "What are the comedy movies released between 2010 and 2020?",
        {
            "query": "",
            "filter": 'and(gte("Year", 2010), lte("Year", 2020), or(eq("Genre1", "Comedy"),eq("Genre2", "Comedy"),eq("Genre3", "Comedy")))',
        },
    ),
    (
        "List movies from the comedy genre where there is death or dead people involved.",
        {
            "query": "death or dead people involved",
            "filter": 'or(eq("Genre1", "Comedy"),eq("Genre2", "Comedy"),eq("Genre3", "Comedy"))',
        },
    ),
    (
        "Summarize the movie plots of Steven Spielberg’s top-rated sci-fi movies.",
        {
            "query": "summarize the movie plots",
            "filter": 'and(or(eq("Genre1", "sci-fi"),eq("Genre2", "sci-fi"),eq("Genre3", "sci-fi")), eq("Director", "Steven Spielberg"), gte("Rating", 8.0))',
        },
    ),
]

constructor_prompt = get_query_constructor_prompt(
    document_content_description,
    metadata_field_info,
    allowed_comparators=allowed_comparators,
    examples=examples,
)
# Create query constructor
output_parser = StructuredQueryOutputParser.from_components()

query_constructor = constructor_prompt | llm | output_parser

retriever = SelfQueryRetriever(
    query_constructor=query_constructor,
    vectorstore=vectordb,
    structured_query_translator=ChromaTranslator(),
    search_kwargs={'k': 15}
)

template = ChatPromptTemplate.from_template(
    """
Use the following context movies that matched the user question. Use the movies below only to answer the user's question.

If you don't know the answer, just say that you don't know, don't try to make up an answer.

----
{context}
----
Question: {question}
Answer:
"""
)

#prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(f"{doc.page_content}\n\nMetadata: {doc.metadata}" for doc in docs)

# Create a chatbot Question & Answer chain from the retriever
rag_chain_from_docs = (
    RunnablePassthrough.assign(
        context=(lambda x: format_docs(x["context"])))
    | template
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

# Perform the retrieval and generate the response
# Streamlit interaction
if st.button("Submit", type="primary"):
    if query:
        #response = retriever.get_relevant_documents(query)
        response = rag_chain_with_source.invoke({query})
        st.write(response['answer'])
