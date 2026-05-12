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
