# 🎮 Gen AI-Powered Conversational Bot with IMDB Dataset

This repository contains a conversational AI bot powered by OpenAI's GPT-4 and embeddings (`gpt-4o-mini` and `text-embedding-3-large`) to answer movie-related questions using the IMDB dataset. The bot is hosted on a **Streamlit UI** for an interactive and user-friendly experience.

---

## 🚀 Features

- **Natural Language Querying**: Ask questions about movies, genres, directors, and ratings effortlessly.  
- **IMDB Data Exploration**: Retrieve movie details such as release dates, ratings, metascores, earnings, and more.  
- **Powerful Search**: Combines LLM capabilities with schema-aware searches for precise answers.  
- **Custom Queries**: Handle structured and unstructured queries such as movie plot summaries and genre-specific data.  
- **Interactive UI**: Built with **Streamlit** for a clean and intuitive user experience.  
- **Vectorized Search**: Utilizes embeddings for semantic search on textual data (e.g., overviews, genres).  

---

## 📂 Dataset

The IMDB dataset contains the following columns:  
- **Poster_Link**: URL to the movie poster.  
- **Series_Title**: Name of the movie.  
- **Released_Year**: Release year.  
- **Certificate**: Movie certification (e.g., PG-13, R).  
- **Runtime**: Total runtime of the movie.  
- **Genre**: Movie genre(s).  
- **IMDB_Rating**: IMDB rating of the movie.  
- **Overview**: Brief story or summary.  
- **Meta_score**: Metacritic score.  
- **Director**: Director of the movie.  
- **Star1, Star2, Star3, Star4**: Leading actors.  
- **No_of_votes**: Number of user votes on IMDB.  
- **Gross**: Box office gross earnings.  

---

## 🛠️ Technologies Used

- **LLM**: OpenAI's `gpt-4o-mini` for conversational interactions.  
- **Embeddings**: `text-embedding-3-large` for semantic vectorization.  
- **Vector Store**: ChromaDB for storing and retrieving vectorized embeddings.  
- **UI Framework**: Streamlit for an intuitive chatbot interface.  

---

## 💡 Sample Questions

Below are some test questions the bot can answer (but is not limited to):  

1. When did *The Godfather* release?  
2. What are the top 5 movies of 2019 by Metascore?  
3. List the top 5 horror movies (2010–2020) by IMDB rating.  
4. Top horror movies with a Metascore above 75 and IMDB rating above 8.  
5. Top directors and their highest-grossing movies (gross > $500M).  

---

## ⚙️ Setup Instructions

### Prerequisites
Ensure you have Python 3.8+ installed.  

### 1⃣ Install Required Libraries
Run the following command to install dependencies:  
```bash  
pip install -r requirements.txt  
```  

### 2⃣ Create the Vector Database
Generate the ChromaDB instance by running:  
```bash  
python df_sql.py  
```  

### 3⃣ Launch the Streamlit App
Start the chatbot UI with:  
```bash  
streamlit run IMDB_movie.py  
```  
