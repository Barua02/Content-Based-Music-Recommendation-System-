# app.py
"""
Streamlit UI for Music Recommendation App
- Loads processed data and exposes interactive recommendation interface
"""

import streamlit as st
from recommendation import df, recommend_songs

# Set custom Streamlit page config
st.set_page_config(
    page_title="Music Recommender 🎵",
    page_icon="🎧",  # You can also use a path to a .ico or .png file
    layout="centered"
)

st.title("🎶 Instant Music Recommender")

# Song selection widget
song_list = sorted(df['song'].dropna().unique())
selected_song = st.selectbox("🎵 Select a song:", song_list)

# Recommendation trigger
if st.button("🚀 Recommend Similar Songs"):
    with st.spinner("Finding similar songs..."):
        recommendations = recommend_songs(selected_song)
        if recommendations is None:
            st.warning("Sorry, song not found.")
        else:
            st.success("Top similar songs:")
            st.table(recommendations)