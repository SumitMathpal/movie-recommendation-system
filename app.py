import os
import pickle
import requests
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Resolve path to .env file relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path)

# Page configuration
st.set_page_config(
    page_title="CineMatch | AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design (Glassmorphism, Dark Theme, Micro-animations)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: linear-gradient(135deg, #070514 0%, #0d0925 50%, #030208 100%) !important;
        font-family: 'Outfit', sans-serif !important;
        color: #f1f1f6 !important;
    }
    
    /* Text overrides */
    h1, h2, h3, h4, h5, h6, p, span, label {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(10, 7, 28, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
    }
    
    /* Glassmorphic card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 20px !important;
        padding: 28px !important;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.45) !important;
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
        margin-bottom: 20px;
    }
    .glass-card:hover {
        transform: translateY(-4px) !important;
        border-color: rgba(139, 92, 246, 0.3) !important;
        box-shadow: 0 15px 45px 0 rgba(139, 92, 246, 0.12) !important;
    }
    
    /* Movie poster grid card styling */
    .movie-card {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
    }
    .movie-card:hover {
        transform: translateY(-8px) scale(1.02) !important;
        border-color: rgba(6, 182, 212, 0.4) !important;
        box-shadow: 0 12px 30px rgba(6, 182, 212, 0.2) !important;
    }
    
    .movie-poster {
        width: 100% !important;
        height: 320px !important;
        object-fit: cover !important;
        transition: transform 0.5s ease !important;
    }
    .movie-card:hover .movie-poster {
        transform: scale(1.05) !important;
    }
    
    .movie-info {
        padding: 16px !important;
        display: flex !important;
        flex-direction: column !important;
        flex-grow: 1 !important;
        justify-content: space-between !important;
        background: linear-gradient(to top, rgba(10, 7, 28, 0.95), rgba(10, 7, 28, 0.5)) !important;
    }
    
    .movie-title {
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        color: #ffffff !important;
        margin-bottom: 8px !important;
        display: -webkit-box !important;
        -webkit-line-clamp: 2 !important;
        -webkit-box-orient: vertical !important;
        overflow: hidden !important;
        height: 2.4em !important;
        line-height: 1.2em !important;
    }
    
    .movie-meta {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        font-size: 0.85rem !important;
        color: #a1a1aa !important;
    }
    
    .rating-badge {
        background: rgba(245, 158, 11, 0.15) !important;
        color: #fbbf24 !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 0.8rem !important;
        border: 1px solid rgba(245, 158, 11, 0.2) !important;
    }
    
    /* Header Gradient and Glow */
    .glow-text {
        font-weight: 800 !important;
        background: linear-gradient(90deg, #c084fc, #67e8f9) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-shadow: 0 0 35px rgba(103, 232, 249, 0.15) !important;
        letter-spacing: -0.5px !important;
    }
    
    /* Styled buttons */
    .stButton > button {
        background: linear-gradient(90deg, #6d28d9 0%, #0891b2 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(109, 40, 217, 0.35) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        text-transform: uppercase !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.5px !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(8, 145, 178, 0.5) !important;
        border: none !important;
        color: #ffffff !important;
    }
    
    /* Smaller card select button */
    .card-btn > .stButton > button {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: none !important;
        padding: 6px 12px !important;
        border-radius: 8px !important;
        text-transform: none !important;
        font-size: 0.8rem !important;
        margin-top: 10px !important;
    }
    .card-btn > .stButton > button:hover {
        background: linear-gradient(90deg, #6d28d9 0%, #0891b2 100%) !important;
        border: none !important;
    }
    
    /* Override standard input borders */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-gradient: linear-gradient(90deg, #6d28d9 0%, #0891b2 100%) !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_resource
def load_movie_data():
    try:
        # Resolve paths relative to the script's directory to avoid CWD issues
        base_dir = os.path.dirname(os.path.abspath(__file__))
        df_path = os.path.join(base_dir, "df.pkl")
        tfidf_path = os.path.join(base_dir, "TFIDF_matrix.pkl")
        indices_path = os.path.join(base_dir, "indices.pkl")

        # Check if files exist
        if not (os.path.exists(df_path) and os.path.exists(tfidf_path) and os.path.exists(indices_path)):
            st.error(f"Dataset pickles not found in workspace! Searched in: {base_dir}")
            return None, None, None
            
        with open(df_path, "rb") as f:
            df = pickle.load(f)
        with open(tfidf_path, "rb") as f:
            tfidf_matrix = pickle.load(f)
        with open(indices_path, "rb") as f:
            indices = pickle.load(f)
            
        # Clean numeric columns to prevent type errors (e.g. mixed float and string values)
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
        df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
        df["title"] = df["title"].astype(str)
            
        return df, tfidf_matrix, indices
    except Exception as e:
        st.error(f"Error loading pickle files: {e}")
        return None, None, None

df, TFIDF_matrix, indices = load_movie_data()

# Fallback poster image from Unsplash
PLACEHOLDER_POSTER = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=500&auto=format&fit=crop"

# Caching TMDB API calls to speed up loading
@st.cache_data
def fetch_movie_details_tmdb(title):
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        return None
        
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={requests.utils.quote(title)}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                # Take the most relevant search result
                movie = data["results"][0]
                poster_path = movie.get("poster_path")
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else PLACEHOLDER_POSTER
                
                movie_id = movie.get("id")
                trailer_url = None
                
                # Fetch videos (trailers) if possible
                if movie_id:
                    v_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={api_key}"
                    v_resp = requests.get(v_url, timeout=3)
                    if v_resp.status_code == 200:
                        v_data = v_resp.json()
                        for vid in v_data.get("results", []):
                            if vid.get("site") == "YouTube" and vid.get("type") in ["Trailer", "Teaser"]:
                                trailer_url = f"https://www.youtube.com/watch?v={vid.get('key')}"
                                break
                                
                return {
                    "id": movie_id,
                    "title": movie.get("title"),
                    "overview": movie.get("overview", "No description available."),
                    "poster_url": poster_url,
                    "rating": movie.get("vote_average", 0.0),
                    "release_date": movie.get("release_date", "Unknown"),
                    "trailer_url": trailer_url
                }
    except Exception:
        pass
    return None

def get_recommendations(title, n=10):
    if df is None or TFIDF_matrix is None or indices is None:
        return []
        
    if title not in indices:
        return []
        
    idx = indices[title]
    
    # Handle duplicate titles that result in Series
    if isinstance(idx, (pd.Series, np.ndarray)):
        idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]
        
    # Calculate cosine similarities
    sim_scores = cosine_similarity(TFIDF_matrix[idx], TFIDF_matrix).flatten()
    
    # Sort in descending order
    similar_indices = sim_scores.argsort()[::-1]
    
    recommendations = []
    seen_titles = {title.lower()}
    
    for i in similar_indices:
        if i == idx:
            continue
            
        rec_title = df["title"].iloc[i]
        rec_title_lower = rec_title.lower()
        
        if rec_title_lower not in seen_titles:
            seen_titles.add(rec_title_lower)
            recommendations.append(rec_title)
            
        if len(recommendations) == n:
            break
            
    return recommendations

# Initialize session state for selected movie
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

# Sidebar layout
with st.sidebar:
    st.markdown("<h2 class='glow-text' style='text-align: center;'>🎬 CineMatch</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a1a1aa; font-size: 0.9rem;'>AI-Powered Movie Recommendations</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### Search Parameters")
    num_recommendations = st.slider("Number of recommendations:", min_value=5, max_value=20, value=10)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    CineMatch uses a Content-Based Filtering algorithm matching movies by keywords, genres, and overviews using **TF-IDF Vectorization** and **Cosine Similarity**. 
    
    Posters, trailers, and ratings are loaded dynamically using the **TMDB API**.
    """)
    
    # Check TMDB API key status
    api_key = os.getenv("TMDB_API_KEY")
    if api_key:
        st.success("TMDB API Key: Active")
    else:
        st.warning("TMDB API Key: Missing! Check .env file.")

# Main Application Layout
st.markdown("<h1 class='glow-text' style='margin-bottom: 5px;'>CineMatch Recommendation Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.15rem; color: #a1a1aa; margin-bottom: 25px;'>Find your next favorite movie instantly using our content-based AI matching algorithm.</p>", unsafe_allow_html=True)

if df is not None:
    # Interactive Movie Search with Auto-filtering
    unique_titles = df["title"].unique()
    
    search_query = st.text_input(
        "🔍 Search for a movie title to get started:", 
        placeholder="Type a movie title... (e.g. Toy Story, Jumanji, Avatar)",
        key="search_input"
    )
    
    # Filter matching movies based on user's query
    if search_query:
        matches = [t for t in unique_titles if search_query.lower() in str(t).lower()]
        if matches:
            selected_from_dropdown = st.selectbox(
                "🎯 Select the movie from search results:", 
                options=matches,
                index=0
            )
            if st.button("Generate Recommendations"):
                st.session_state.selected_movie = selected_from_dropdown
        else:
            st.error("No movies found matching your search. Please try another title!")
            selected_from_dropdown = None
    else:
        # Default view with top popular movies from dataset
        popular_movies = df.sort_values(by="popularity", ascending=False)["title"].head(100).unique()
        selected_from_dropdown = st.selectbox(
            "🎯 Select or search from popular movies:",
            options=["Select a movie..."] + list(popular_movies),
            index=0
        )
        if selected_from_dropdown != "Select a movie...":
            if st.button("Generate Recommendations"):
                st.session_state.selected_movie = selected_from_dropdown

    # If a movie is selected, show details and recommendations
    if st.session_state.selected_movie:
        current_movie = st.session_state.selected_movie
        
        st.markdown("---")
        st.markdown(f"## 🎬 Currently Selected: **{current_movie}**")
        
        # Fetch TMDB Details for the currently selected movie
        details = fetch_movie_details_tmdb(current_movie)
        
        # Display selected movie details in a card
        with st.container():
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 3])
            
            with col1:
                poster_url = details["poster_url"] if details else PLACEHOLDER_POSTER
                st.image(poster_url, use_container_width=True)
                
            with col2:
                if details:
                    st.markdown(f"<h3 style='margin-top: 0;'>{details['title']}</h3>", unsafe_allow_html=True)
                    
                    # Metadata badges
                    rating = details['rating']
                    release_date = details['release_date']
                    st.markdown(
                        f"<p><span class='rating-badge'>★ {rating:.1f}/10</span> &nbsp;&nbsp; "
                        f"📅 <b>Release Date:</b> {release_date}</p>",
                        unsafe_allow_html=True
                    )
                    
                    # Display original genres from dataset if available
                    movie_row = df[df["title"] == current_movie]
                    if not movie_row.empty:
                        genres = movie_row.iloc[0]["genres"]
                        if genres:
                            st.markdown(f"🏷️ <b>Genres:</b> {genres}")
                            
                    st.markdown("#### Overview")
                    st.markdown(details['overview'])
                    
                    # Display YouTube Trailer if available
                    if details['trailer_url']:
                        st.markdown("#### Movie Trailer")
                        st.video(details['trailer_url'])
                else:
                    # Fallback dataset values if TMDB lookup fails
                    st.markdown(f"<h3 style='margin-top: 0;'>{current_movie}</h3>", unsafe_allow_html=True)
                    movie_row = df[df["title"] == current_movie]
                    if not movie_row.empty:
                        overview = movie_row.iloc[0]["overview"]
                        genres = movie_row.iloc[0]["genres"]
                        rating = movie_row.iloc[0]["vote_average"]
                        
                        st.markdown(f"<p><span class='rating-badge'>★ {rating:.1f}/10</span></p>", unsafe_allow_html=True)
                        if genres:
                            st.markdown(f"🏷️ <b>Genres:</b> {genres}")
                        st.markdown("#### Overview")
                        st.markdown(overview if overview else "No description available.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Recommendation generation and rendering
        with st.spinner("Analyzing contents and fetching recommendation details..."):
            recommended_titles = get_recommendations(current_movie, n=num_recommendations)
            
        if recommended_titles:
            st.markdown(f"## 🚀 Top {len(recommended_titles)} Recommendations for You")
            
            # Fetch recommended movie details
            rec_details_list = []
            for rec_title in recommended_titles:
                rec_det = fetch_movie_details_tmdb(rec_title)
                if rec_det:
                    rec_details_list.append(rec_det)
                else:
                    # Fallback info if TMDB fails
                    rec_row = df[df["title"] == rec_title]
                    vote_avg = rec_row.iloc[0]["vote_average"] if not rec_row.empty else 0.0
                    rec_details_list.append({
                        "title": rec_title,
                        "poster_url": PLACEHOLDER_POSTER,
                        "rating": vote_avg,
                        "release_date": "N/A"
                    })
            
            # Render recommendations in a grid
            cols_per_row = 5
            for row_idx in range(0, len(rec_details_list), cols_per_row):
                row_movies = rec_details_list[row_idx:row_idx+cols_per_row]
                cols = st.columns(cols_per_row)
                
                for col_idx, movie in enumerate(row_movies):
                    with cols[col_idx]:
                        # Wrap each movie card in a customized styled container
                        st.markdown(
                            f"""
                            <div class='movie-card'>
                                <img src='{movie["poster_url"]}' class='movie-poster'>
                                <div class='movie-info'>
                                    <div class='movie-title' title='{movie["title"]}'>{movie["title"]}</div>
                                    <div class='movie-meta'>
                                        <span class='rating-badge'>★ {movie["rating"]:.1f}</span>
                                        <span>{movie["release_date"][:4] if movie["release_date"] != 'N/A' else 'N/A'}</span>
                                    </div>
                                </div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # Add a clean select button to re-run recommendations on the clicked movie
                        # Use a unique key for the button to avoid duplication errors
                        st.markdown("<div class='card-btn'>", unsafe_allow_html=True)
                        if st.button("Explore Similar", key=f"btn_{movie['title']}_{row_idx}_{col_idx}"):
                            st.session_state.selected_movie = movie["title"]
                            st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No recommendations found for this movie. Try choosing another one.")
else:
    st.error("Error: Movie dataset could not be loaded. Please verify the dataset pickle file path.")
