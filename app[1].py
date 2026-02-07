delete app[1].py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

@st.cache_data
def load_data():
    movies = pd.read_csv("movies[1].csv")
    tags = pd.read_csv("tags[1].csv")
    links = pd.read_csv("links[1].csv")

    tag_data = tags.groupby('movieId')['tag'].apply(lambda x: " ".join(x)).reset_index()
    movies = movies.merge(tag_data, on="movieId", how="left")
    movies["tag"] = movies["tag"].fillna("")

    movies = movies.merge(links, on="movieId", how="left")
    return movies

movies = load_data()

@st.cache_resource
def create_similarity():
    movies["features"] = movies["genres"] + " " + movies["tag"]
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = create_similarity()

def find_similar_movies(movie_name, num_recommendations=5):
    movie_name = movie_name.lower()
    all_titles = movies["title"].str.lower().tolist()
    matches = get_close_matches(movie_name, all_titles, n=1, cutoff=0.4)
    if not matches:
        return None, None
    matched_title = movies[movies["title"].str.lower() == matches[0]]["title"].values[0]
    idx = indices[matched_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return matched_title, movies.iloc[movie_indices]

movie_input = st.text_input("Enter Movie Name")
num_rec = st.slider("Number of Recommendations", 1, 10, 5)

if st.button("Recommend"):
    title, recommendations = find_similar_movies(movie_input, num_rec)
    if title is None:
        st.warning("Movie not found")
    else:
        st.subheader(f"Movies similar to: {title}")
        for _, row in recommendations.iterrows():
            st.markdown(f"**{row['title']}**")
            st.write(row["genres"])
