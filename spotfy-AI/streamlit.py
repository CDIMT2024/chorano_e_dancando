import streamlit as st
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import os
from dotenv import load_dotenv
from utils import get_rf
import pandas as pd
load_dotenv()
# Set up Spotipy with Spotify API credentials
client_id = os.getenv('CLIENT_ID0')
client_secret = os.getenv('CLIENT_SECRET0')
client_credentials_manager = SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Streamlit app title
st.title("Spotify Music Search")

# Search bar for track name
search_query = st.text_input("Search for a track:")

if search_query:
    # Search for tracks with Spotipy
    results = sp.search(q=search_query, type='track', limit=10)
    
    # Extract track names and IDs
    tracks = {track['name']: track['id'] for track in results['tracks']['items']}
    
    # Display track selection
    track_name = st.selectbox("Select a track:", list(tracks.keys()))
    
    if track_name:
        # Get the selected track's ID
        track_id = tracks[track_name]
        
        # Fetch audio features for the selected track
        audio_features = sp.audio_features(track_id)[0]
        
        # Display audio features
        if audio_features:
            st.subheader(f"Audio Features for '{track_name}'")
            st.write(audio_features)
            df = pd.DataFrame([audio_features])
            st.write(df)
            # Predict the genre of the track
            model = get_rf()
            df = df.drop(
                columns=['id', 'uri', 'track_href', 'analysis_url', 'type', 'time_signature']
            )
            genre = model.predict(df)[0]
            st.write(f"The predicted genre is: {genre}")

