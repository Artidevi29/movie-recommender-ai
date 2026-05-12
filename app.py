# app.py - Movie Recommender System (Updated for ZIP files)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import ast

# File paths for the ZIP archives
MOVIES_PATH = "tmdb_5000_movies.zip"
CREDITS_PATH = "tmdb_5000_credits.zip"

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

@st.cache_data
def load_and_process():
    # Pandas can read CSV files directly from a ZIP archive
    movies = pd.read_csv(MOVIES_PATH)
    credits = pd.read_csv(CREDITS_PATH)

    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords']]
    movies.dropna(inplace=True)

    def convert(obj):
        if pd.isna(obj):
            return []
        try:
            L = []
            for i in ast.literal_eval(obj):
                L.append(i['name'])
            return L
        except:
            return []

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['tags'] = movies['overview'].fillna('') + ' ' + \
                     movies['genres'].apply(lambda x: ' '.join(x)) + ' ' + \
                     movies['keywords'].apply(lambda x: ' '.join(x))

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return movies, similarity

st.title("🎬 AI Movie Recommendation System")

movies, similarity = load_and_process()

def recommend(movie_title):
    idx = movies[movies['title'] == movie_title].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    return [movies.iloc[i[0]].title for i in distances[1:6]]

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie you like:", movie_list)

if st.button("🎯 Recommend Similar Movies"):
    with st.spinner("Finding recommendations..."):
        recs = recommend(selected_movie)
        st.success("✅ Here are your recommendations:")
        for i, movie in enumerate(recs, 1):
            st.write(f"{i}. {movie}")
