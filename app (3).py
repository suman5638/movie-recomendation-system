
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

OMDB_API_KEY = "eae729f3" # Replace with your actual OMDB API key


# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
tags = pd.read_csv("tags.csv")
links =pd.read_csv("links.csv")

# Merge imdbId into movies DataFrame
movies = movies.merge(links[['movieId', 'imdbId']], on='movieId', how='left')

# Combine all tags of each movie
tags_grouped = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()

# Merge tags into movies
movies = movies.merge(tags_grouped, on="movieId", how="left")

# Fill NaN tags with empty string
movies["tag"] = movies["tag"].fillna("")

# Clean genres: replace '|' with space
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

# Combine genres + tags for metadata
movies["metadata"] = movies["genres"] + " " + movies["tag"]

# --- Featured Movie for Hero Section ---
featured_movie_title_hero = "Inception (2010)"
featured_movie_data_hero = movies[movies['title'] == featured_movie_title_hero].iloc[0]
featured_movie_genres_hero = featured_movie_data_hero['genres']
featured_movie_tag_hero = featured_movie_data_hero['tag']
featured_movie_imdb_id_hero = featured_movie_data_hero['imdbId']
featured_movie_poster_url_hero = get_poster(featured_movie_imdb_id_hero)
# Creating a short description for the hero section
featured_movie_description_hero = f"Action, Sci-Fi, Thriller. Director: Christopher Nolan. Starring: Leonardo DiCaprio. A thief who steals corporate secrets through use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."


# Content-based Filtering Setup
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["metadata"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# Content-based recommendation function
def content_recommend(title, n=10):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres', 'tag', 'imdbId']]

# Collaborative Filtering Setup
# Use original ratings for the main model, not the split one for evaluation
user_item_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

# Apply SVD
matrix = user_item_matrix.values
U, sigma, Vt = svds(matrix, k=50) # Using k=50 for the deployed model
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_df = pd.DataFrame(
    predicted_ratings,
    columns=user_item_matrix.columns,
    index=user_item_matrix.index
)

# Collaborative filtering recommendation function
def collaborative_recommend(user_id, n=10):
    user_row = predicted_df.loc[user_id]
    already_rated = ratings[ratings.userId == user_id]['movieId']
    recommendations = user_row.drop(already_rated, errors='ignore')
    top_movies = recommendations.sort_values(ascending=False).head(n)
    return movies[movies.movieId.isin(top_movies.index)][['movieId', 'title', 'genres', 'tag', 'imdbId']]

# Hybrid recommendation function
def hybrid_recommend(user_id, title, n=10, alpha=0.6):
    idx = indices[title]
    sim_scores_content = list(enumerate(cosine_sim[idx]))

    hybrid_scores = []

    for i, sim_score in sim_scores_content:
        movie_id = movies.iloc[i]["movieId"]

        collab_score = 0
        if user_id in predicted_df.index and movie_id in predicted_df.columns:
            collab_score = predicted_df.loc[user_id, movie_id]

        final_score = alpha * sim_score[1] + (1 - alpha) * collab_score

        hybrid_scores.append((i, final_score))

    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    hybrid_scores = hybrid_scores[1:n+1]

    movie_indices = [i[0] for i in hybrid_scores]

    return movies.iloc[movie_indices][["title", "genres", "tag", "imdbId"]]

# Function to get movie poster from OMDb API
def get_poster(imdb_id):
    if pd.isna(imdb_id) or imdb_id == '':
        return "https://via.placeholder.com/200x300.png?text=No+Poster"
    try:
        # Ensure imdb_id is prefixed with 'tt' and formatted correctly for the API
        if not str(imdb_id).startswith('tt'):
            imdb_id = f"tt{int(imdb_id):07d}"
        else:
            imdb_id = str(imdb_id)

        url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
        response = requests.get(url)
        data = response.json()
        if data and data.get("Poster") and data["Poster"] != "N/A":
            return data["Poster"]
        else:
            return "https://via.placeholder.com/200x300.png?text=No+Poster"
    except Exception as e:
        st.error(f"Error fetching poster for {imdb_id}: {e}")
        return "https://via.placeholder.com/200x300.png?text=No+Poster"

# Function for finding similar movies for Streamlit display
def find_similar_movies_streamlit(movie_name, num_recommendations=10):
    movie_name_lower = movie_name.lower()
    base_movie_title = None

    # 1. Try to find an exact match
    exact_match = movies[movies['title'].str.lower() == movie_name_lower]
    if not exact_match.empty:
        base_movie_title = exact_match.iloc[0]['title']
    else:
        # 2. If no exact match, find titles that start with the query
        starts_with_matches = movies[movies['title'].str.lower().str.startswith(movie_name_lower)]
        if not starts_with_matches.empty:
            # Prioritize shorter titles if multiple movies start with the query
            base_movie_title = starts_with_matches.loc[starts_with_matches['title'].str.len().idxmin()]['title']
        else:
            # 3. If no 'starts with' match, find titles that contain the query
            contains_matches = movies[movies['title'].str.lower().str.contains(movie_name_lower)]
            if not contains_matches.empty:
                # Prioritize shorter titles if multiple movies contain the query
                base_movie_title = contains_matches.loc[contains_matches['title'].str.len().idxmin()]['title']

    if base_movie_title is None or base_movie_title not in indices:
        return pd.DataFrame(), None # No suitable movie found or not in content-based index

    idx = indices[base_movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]

    movie_indices = [i[0] for i in sim_scores]

    return movies.iloc[movie_indices][['title','genres','tag', 'imdbId']], base_movie_title


# Streamlit UI
st.set_page_config(layout="wide")
st.markdown(
    f'''
    <style>
        .reportview-container .main .block-container{{
            max-width: 1200px;
            padding-top: 2rem;
            padding-right: 2rem;
            padding-left: 2rem;
            padding-bottom: 2rem;
        }}
        body {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #e5e5e5;
            background-image: url("https://images.unsplash.com/photo-1579547621113-e4bb2a19ff62?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D") !important; /* A 3D modern-style background */
            background-size: cover !important;
            background-attachment: fixed !important; /* Keeps background fixed when scrolling */
            background-color: #141414 !important; /* Dark background */
        }
        .full-width-image {{
            width: 100%;
            height: 250px; /* Increased height for better visual impact */
            object-fit: cover;
            border-radius: 4px; /* Slightly reduced border-radius */
            margin-bottom: 8px;
        }}
        .movie-card {{
            border: 1px solid #222; /* Darker border */
            background-color: #1a1a1a; /* Slightly darker background for cards */
            border-radius: 6px; /* Slightly reduced border-radius */
            padding: 12px;
            margin-bottom: 15px;
            height: 420px; /* Adjusted fixed height for movie card */
            display: flex;
            flex-direction: column;
            justify-content: flex-start; /* Align content to the top */
            overflow: hidden;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5); /* Stronger shadow for depth */
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }}
        .movie-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.7);
        }}
        .movie-title {{
            font-size: 1.2em; /* Slightly larger title */
            font-weight: bold;
            min-height: 60px; /* Increased ensure title space */
            color: white;
            text-align: left;
            margin-bottom: 5px;
        }}
        .movie-details {{
            font-size: 0.85em; /* Slightly smaller details font */
            color: #ccc; /* Lighter grey for better contrast */
            flex-grow: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            line-height: 1.4;
            margin-top: 5px;
        }}
        .movie-details strong {{
            color: #f0f0f0; /* Ensure strong labels are visible */
        }}
        .stButton>button {{
            width: 100%;
            background-color: #e50914; /* Netflix red button */
            color: white;
            border-radius: 4px;
            border: none;
            padding: 10px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }}
        .stButton>button:hover {
            background-color: #ff0000;
        }
        .hero-section {
            position: relative;
            height: 450px; /* Adjust height as needed */
            background-size: cover;
            background-position: center;
            display: flex;
            align-items: center;
            color: white;
            padding: 0 50px;
            margin-bottom: 30px;
            border-radius: 8px;
            overflow: hidden;
        }
        .hero-content {
            max-width: 50%;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        }
        .hero-title {
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .hero-description {
            font-size: 1.2em;
            margin-bottom: 20px;
            line-height: 1.5;
        }
        .hero-buttons .stButton>button {
            display: inline-block;
            width: auto;
            margin-right: 15px;
            padding: 10px 25px;
            font-size: 1.1em;
            cursor: pointer;
        }
        .hero-buttons .primary-button {
            background-color: white;
            color: black;
        }
        .hero-buttons .primary-button:hover {
            background-color: rgba(255,255,255,0.8);
        }
        .hero-buttons .secondary-button {
            background-color: rgba(109,109,110,0.7);
            color: white;
        }
        .hero-buttons .secondary-button:hover {
            background-color: rgba(109,109,110,0.9);
        }
        .header-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 20px;
            background-color: #141414; /* Dark background for header */
            color: white;
            border-bottom: 1px solid #333;
            margin-bottom: 20px;
        }
        .logo {
            font-size: 2em;
            font-weight: bold;
            color: #E50914; /* Netflix red */
        }
        .nav-links {
            display: flex;
            gap: 20px;
        }
        .nav-item {
            color: white;
            text-decoration: none;
            font-size: 1.1em;
            transition: color 0.3s ease;
        }
        .nav-item:hover {
            color: #E50914;
        }

        .scrollable-row-container {
            display: flex;
            overflow-x: auto; /* Enable horizontal scrolling */
            padding-bottom: 20px; /* Space for scrollbar */
            scrollbar-width: thin; /* Firefox */
            scrollbar-color: #e50914 #1e1e1e; /* Firefox scrollbar color */
        }
        /* Webkit scrollbar styling */
        .scrollable-row-container::-webkit-scrollbar {
            height: 8px;
        }
        .scrollable-row-container::-webkit-scrollbar-track {
            background: #1e1e1e;
            border-radius: 10px;
        }
        .scrollable-row-container::-webkit-scrollbar-thumb {
            background: #e50914;
            border-radius: 10px;
        }
        .movie-card-wrapper {
            flex: 0 0 auto; /* Prevent cards from shrinking */
            width: 200px; /* Fixed width for each card */
            margin-right: 15px; /* Space between cards */
        }

        .stTextInput > div > div > input {
            background-color: #333;
            color: white;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 10px;
            font-size: 1.1em;
        }
        .stTextInput label {
            color: white;
            font-size: 1.2em;
            margin-bottom: 10px;
        }

    </style>
    '''
,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="header-container">
        <div class="logo">MovieFlix</div>
        <div class="nav-links">
            <a href="#" class="nav-item">Home</a>
            <a href="#" class="nav-item">TV Shows</a>
            <a href="#" class="nav-item">Movies</a>
            <a href="#" class="nav-item">New & Popular</a>
            <a href="#" class="nav-item">My List</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Global Search Bar
st.markdown("<h2 style='text-align: center; color: white; margin-top: 20px; margin-bottom: 20px;'>Find Your Next Favorite Movie</h2>", unsafe_allow_html=True)
global_search_query = st.text_input("Search for movies:", "", key="global_search_input", help="Enter a movie title to search globally.")
if st.button("Search All Movies", key="global_search_button"):
    if global_search_query:
        search_results, base_title_for_display = find_similar_movies_streamlit(global_search_query)
        if not search_results.empty:
            st.subheader(f"Search Results for: {base_title_for_display}")
            display_recommendations(search_results)
        else:
            st.write("No movies found matching your search.")
    else:
        st.write("Please enter a movie title to search.")

st.markdown(
    f"""
    <div class="hero-section" style="background-image: linear-gradient(to top, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0) 60%, rgba(0,0,0,0.8) 100%), url({featured_movie_poster_url_hero});">
        <div class="hero-content">
            <h1 class="hero-title">{featured_movie_title_hero}</h1>
            <p class="hero-description">{featured_movie_description_hero}</p>
            <div class="hero-buttons">
                <button class="stButton primary-button">▶ Play</button>
                <button class="stButton secondary-button">ⓘ More Info</button>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Content-Based", "Collaborative Filtering", "Hybrid Recommendation", "Find Similar Movies"])

# Helper function to display movie recommendations with posters in a scrollable row
def display_recommendations(movie_list):
    if isinstance(movie_list, pd.DataFrame) and movie_list.empty:
        st.write("No recommendations found.")
        return
    if movie_list.empty:
        st.write("No recommendations found.")
        return

    num_movies = len(movie_list)
    if num_movies == 0:
        st.write("No recommendations found.")
        return

    movie_cards_html = ""
    for idx in range(num_movies):
        movie = movie_list.iloc[idx]
        title = movie['title']
        genres = movie['genres']
        tag = movie['tag']
        imdb_id = movie['imdbId']
        poster_url = get_poster(imdb_id)
        
        movie_cards_html += f'''
            <div class="movie-card-wrapper">
                <div class="movie-card">
                    <div>
                        <img src="{poster_url}" class="full-width-image" onerror="this.onerror=null;this.src='https://via.placeholder.com/200x300.png?text=No+Poster';"/>
                        <div class="movie-title">{title}</div>
                        <div class="movie-details">
                            <strong>Genres:</strong> {genres}<br>
                            <strong>Tags:</strong> {tag}
                        </div>
                    </div>
                </div>
            </div>
        '''

    st.markdown(
        f'''
        <div class="scrollable-row-container">
            {movie_cards_html}
        </div>
        ''',
        unsafe_allow_html=True
    )

# --- Main UI Layout --- #
# Removed the previous if/elif structure for page navigation,
# as the hero section and subsequent recommendation rows will be displayed directly.
# Navigation will now use st.sidebar.radio to control which content appears below the hero section.

if page == "Content-Based":
    st.header("Content-Based Recommendations")
    movie_title = st.selectbox("Select a movie:", movies['title'].sort_values().unique(), key="cb_movie_select")
    if st.button("Get Content-Based Recommendations", key="cb_button"):
        if movie_title:
            recommendations = content_recommend(movie_title)
            display_recommendations(recommendations)
        else:
            st.write("Please select a movie.")

elif page == "Collaborative Filtering":
    st.header("Collaborative Filtering Recommendations")
    available_user_ids = sorted(ratings['userId'].unique())
    user_id = st.selectbox("Select a User ID:", available_user_ids, key="cf_user_select")
    if st.button("Get Collaborative Recommendations", key="cf_button"):
        if user_id:
            recommendations = collaborative_recommend(user_id)
            display_recommendations(recommendations)
        else:
            st.write("Please select a User ID.")

elif page == "Hybrid Recommendation":
    st.header("Hybrid Recommendations")
    available_user_ids = sorted(ratings['userId'].unique())
    user_id_hybrid = st.selectbox("Select a User ID for Hybrid:", available_user_ids, key="hybrid_user_select")
    movie_title_hybrid = st.selectbox("Select a movie as a starting point:", movies['title'].sort_values().unique(), key="hybrid_movie_select")
    alpha_val = st.slider("Adjust Alpha (Content vs. Collaborative weight):", 0.0, 1.0, 0.6, 0.1, key="hybrid_alpha_slider")
    if st.button("Get Hybrid Recommendations", key="hybrid_button"):
        if user_id_hybrid and movie_title_hybrid:
            recommendations = hybrid_recommend(user_id_hybrid, movie_title_hybrid, alpha=alpha_val)
            display_recommendations(recommendations)
        else:
            st.write("Please select both a User ID and a movie.")

elif page == "Find Similar Movies":
    st.header("Find Similar Movies")
    search_query = st.text_input("Enter a movie title (or part of it): ", key="find_similar_search_query")
    if st.button("Search for Similar Movies", key="find_similar_button"):
        if search_query:
            # Re-implement find_similar_movies to return DataFrame for display_recommendations
            def find_similar_movies_streamlit(movie_name, num_recommendations=10):
                movie_name = movie_name.lower()
                matches = movies[movies['title'].str.lower().str.contains(movie_name)]

                if matches.empty:
                    return pd.DataFrame() # Return empty DataFrame

                title = matches.iloc[0]['title']
                
                # Check if the title is in the indices, which might not be the case after filtering movies
                if title not in indices:
                    return pd.DataFrame()

                idx = indices[title]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:num_recommendations+1]
                movie_indices = [i[0] for i in sim_scores]
                
                return movies.iloc[movie_indices][['title', 'genres', 'tag', 'imdbId']]

            recommendations = find_similar_movies_streamlit(search_query)
            if not recommendations.empty:
                st.subheader(f"Movies similar to: {base_title_for_display}")
                display_recommendations(recommendations)
            else:
                st.write("No similar movies found for your query.")
        else:
            st.write("Please enter a movie title to search.")
