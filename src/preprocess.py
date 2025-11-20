# preprocess.py
"""
Preprocesses the raw music dataset for recommendation.
- Cleans lyrics
- Vectorizes with TF-IDF
- Saves processed data and matrix for downstream use
"""
import pandas as pd
import re
import nltk
import joblib
import logging
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
DATA_PATH = Path("data/spotify_millsongdata.csv")
PKL_DIR = Path("./pkl")
LOG_PATH = Path("./logs/preprocess.log")
PKL_DIR.mkdir(exist_ok=True)
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

logging.info("🚀 Starting preprocessing...")

# Download NLTK resources if needed
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load and sample dataset
try:
    df = pd.read_csv(DATA_PATH)
    logging.info(f"✅ Dataset loaded: {len(df)} rows.")
    # Downsample for performance (optional)
    if len(df) > 10000:
        df = df.sample(10000, random_state=42).reset_index(drop=True)
        logging.info(f"✅ Downsampled to {len(df)} rows.")
except Exception as e:
    logging.error(f"❌ Failed to load dataset: {e}")
    raise

# Drop unused columns
if 'link' in df.columns:
    df = df.drop(columns=['link'])

# Text cleaning setup
stop_words = set(stopwords.words('english'))
stop_words.update({"yeah", "oh", "na"})

def preprocess_text(text):
    """Clean and tokenize lyrics text."""
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

logging.info("🧹 Cleaning lyrics text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)
logging.info("✅ Lyrics cleaned.")

# TF-IDF vectorization
logging.info("🔠 Vectorizing lyrics with TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
logging.info(f"✅ TF-IDF matrix shape: {tfidf_matrix.shape}")

# Save processed data and matrix
joblib.dump(df, PKL_DIR / 'df_cleaned.pkl')
joblib.dump(tfidf_matrix, PKL_DIR / 'tfidf_matrix.pkl')
logging.info("💾 Data saved to pkl/ directory.")
logging.info("✅ Preprocessing complete.")