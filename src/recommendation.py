# recommendation.py
"""
Music Recommendation Engine
- Loads processed data and TF-IDF matrix
- Fits NearestNeighbors model
- Provides recommend_songs function for querying similar tracks
"""
import joblib
import logging
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

PKL_DIR = Path("./pkl")
LOG_PATH = Path("./logs/recommend.log")
LOG_PATH.parent.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🔁 Loading processed data...")
try:
    df = joblib.load(PKL_DIR / 'df_cleaned.pkl')
    tfidf_matrix = joblib.load(PKL_DIR / 'tfidf_matrix.pkl')
    logging.info("✅ Data loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load required files: {e}")
    raise

# Fit NearestNeighbors model for fast similarity search
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(tfidf_matrix)

def recommend_songs(song_name, top_n=5):
    """
    Recommend similar songs based on lyrics similarity.
    Args:
        song_name (str): Song title to find recommendations for.
        top_n (int): Number of recommendations to return.
    Returns:
        pd.DataFrame: DataFrame of recommended songs with similarity scores.
    """
    logging.info(f"🎵 Recommending songs for: '{song_name}'")
    idx = df[df['song'].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        logging.warning("⚠️ Song not found in dataset.")
        return None
    idx = idx[0]
    distances, indices = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n+1)
    indices = indices.ravel()[1:]
    distances = distances.ravel()[1:]
    result_df = df[['artist', 'song']].iloc[indices].reset_index(drop=True)
    result_df['similarity'] = 1 - distances
    result_df.index = result_df.index + 1
    result_df.index.name = "S.No."
    return result_df