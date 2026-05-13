# 🎬 AI Movie Recommendation System

An AI-powered movie recommender that suggests similar movies based on content (genres, keywords, and overview).

## How it works
- Uses **Cosine Similarity** to find movies with similar content
- Processes movie metadata including genres, keywords, and descriptions
- Built with **Streamlit** for the web interface

## Tech Stack
- Python
- Pandas & NumPy
- Scikit-learn
- Streamlit

## Dataset
TMDB 5000 Movie Dataset from Kaggle

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py

<img width="1049" height="868" alt="image" src="https://github.com/user-attachments/assets/fcd85051-9aa7-4af1-bac3-7ff6182b33bc" />

LIVE DEMO
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movie-recommender-ai-utdexazwy3pk33lkmbzjvf.streamlit.app/)
