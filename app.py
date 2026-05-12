# app.py - Movie Recommender System
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import ast

# CHANGE THIS PATH to where you saved the files
MOVIES_PATH = r"C:\Users\khatr\movie_recommender\tmdb_5000_movies.csv"
CREDITS_PATH = r"C:\Users\khatr\movie_recommender\tmdb_5000_credits.csv"

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv(MOVIES_PATH)
    credits = pd.read_csv(CREDITS_PATH)
    return movies, credits

st.title("🎬 AI Movie Recommendation System")

try:
    movies, credits = load_data()
    
    # Merge datasets
    movies = movies.merge(credits, on='title')
    
    # Keep important columns
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords']]
    
    # Drop missing values
    movies.dropna(inplace=True)
    
    # Convert genres and keywords
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
    
    # Combine features
    movies['tags'] = movies['overview'].fillna('') + ' ' + \
                     movies['genres'].apply(lambda x: ' '.join(x)) + ' ' + \
                     movies['keywords'].apply(lambda x: ' '.join(x))
    
    # Create vectors
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    
    # Calculate similarity
    similarity = cosine_similarity(vectors)
    
    # Recommendation function
    def recommend(movie_title):
        idx = movies[movies['title'] == movie_title].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
        
        recommendations = []
        for i in distances[1:6]:
            recommendations.append(movies.iloc[i[0]].title)
        return recommendations
    
    # UI
    movie_list = movies['title'].values
    selected_movie = st.selectbox("Select a movie you like:", movie_list)
    
    if st.button("🎯 Recommend Similar Movies"):
        with st.spinner("Finding recommendations..."):
            recs = recommend(selected_movie)
            st.success("✅ Here are your recommendations:")
            for i, movie in enumerate(recs, 1):
                st.write(f"{i}. {movie}")
                
except FileNotFoundError as e:
    st.error(f"❌ File not found! Please check the paths.\n\nMake sure you have:\n1. Downloaded tmdb_5000_movies.csv\n2. Downloaded tmdb_5000_credits.csv\n3. Updated the paths in the code\n\nError: {e}")
except Exception as e:
    st.error(f"❌ An error occurred: {e}")