import sqlite3
import pandas as pd
from PIL.ImageOps import expand
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
#from db import connect
from langchain_chroma import Chroma
import os
df = pd.read_csv("imdb_top_1000.csv")
df.drop(['Poster_Link'],axis=1,inplace=True)
os.environ['OPENAI_API_KEY'] = 'you_api_key_here'

df.rename(columns={"Released_Year": "Year", "Meta_score": "MetaScore", "No_of_Votes": "Votes", "IMDB_Rating": "Rating","Series_Title": "Title"},inplace=True)

print(df.columns)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0)
df['Year'] = df['Year'].astype(int)
df['MetaScore'] = pd.to_numeric(df['MetaScore'], errors='coerce').fillna(0)
df['MetaScore'] = df['MetaScore'].astype(int)
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce').fillna(0)
df['Votes'] = df['Votes'].astype(int)
df['Gross'] = df['Gross'].str.replace('[\$,€¥]', '', regex=True).str.replace(',', '')
df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce').fillna(0)
df['Gross'] = df['Gross'].astype(int)
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(0.0)
df['Rating'] = df['Rating'].astype(float)

genre_split = df['Genre'].str.split(',',n=3,expand=True)
print(genre_split)
# Rename the first four columns to Genre1, Genre2, Genre3, and Genre4
genre_split.columns = ["Genre1", "Genre2", "Genre3"]
genre_split = genre_split.fillna("")
# Drop the 'Remaining' column if you want to ignore extra genres
#genre_split = genre_split.drop(columns=["Remaining"])

# Combine the split columns with the original DataFrame
df = pd.concat([df, genre_split], axis=1)



print(df.columns)
print(df.dtypes)
#embeddings = OpenAIEmbeddings().embed_documents(df["Overview"].tolist())
metadata_columns = [col for col in df.columns if col != "Overview"]
documents = [
    Document(page_content=row['Overview'], metadata={col: row[col] for col in metadata_columns})
    for _, row in df.iterrows()
]
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory = "./chromdb")