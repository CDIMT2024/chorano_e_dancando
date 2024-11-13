import streamlit as st
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import os
from utils import get_rf
import pandas as pd

# Set up Spotipy with Spotify API credentials
client_id = os.getenv("CLIENT_ID0")
client_secret = os.getenv("CLIENT_SECRET0")
client_credentials_manager = SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Spotify logo URL
spotify_logo_url = "https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg"  # Replace with your desired logo URL or local path

# Streamlit app title with Spotify logo
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="{spotify_logo_url}" alt="Spotify Logo" style="width: 40px; margin-right: 10px;">
        <h1 style="display: inline;">Spotify Music Classifier</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Genre styles with colors and emojis
genre_styles = {
    "brazilian funk": {"color": "#FFD700", "emoji": "🎶"},  # Yellow
    "chill": {"color": "#ADD8E6", "emoji": "😌"},  # Light Blue
    "christian & gospel": {"color": "#6A5ACD", "emoji": "🙏"},  # Slate Blue
    "dance/electronic": {"color": "#FF4500", "emoji": "🎧"},  # Orange Red
    "rock": {"color": "#8B0000", "emoji": "🎸"},  # Dark Red
}


# Function to calculate text color based on background luminance
def get_contrasting_text_color(bg_color):
    bg_color = bg_color.lstrip("#")
    r, g, b = int(bg_color[0:2], 16), int(bg_color[2:4], 16), int(bg_color[4:6], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (
        "#FFFFFF" if luminance < 128 else "#000000"
    )  # Return white for dark backgrounds, black for light


# Search bar for track name
search_query = st.text_input("Search for a track:")

if search_query:
    # Placeholder for track selection loading
    track_placeholder = st.empty()
    with track_placeholder.container():
        st.write("🔍 Searching for tracks...")

    # Search for tracks with Spotipy
    results = sp.search(q=search_query, type="track", limit=10)

    # Extract track names, IDs, and artists
    tracks = {
        f"{track['name']} - {', '.join([artist['name'] for artist in track['artists']])}": track[
            "id"
        ]
        for track in results["tracks"]["items"]
    }

    # Replace loading skeleton with track selection dropdown
    track_placeholder.empty()
    track_name = st.selectbox("Select a track:", list(tracks.keys()))

    if track_name:
        # Placeholder for audio features and prediction loading
        features_placeholder = st.empty()

        with features_placeholder.container():
            st.write("⏳ Loading audio features...")

        # Get the selected track's ID
        track_id = tracks[track_name]

        # Fetch audio features for the selected track
        audio_features = sp.audio_features(track_id)[0]

        # Display audio features in a collapsible section
        if audio_features:
            # Replace loading skeleton with actual audio features
            features_placeholder.empty()
            with st.expander(f"Audio Features for '{track_name}'", expanded=False):
                st.write(audio_features)
                df = pd.DataFrame([audio_features])

            # Placeholder for genre prediction loading
            genre_placeholder = st.empty()
            with genre_placeholder.container():
                st.write("🎶 Predicting genre...")

            # Predict the genre of the track
            model = get_rf()
            df = df.drop(
                columns=[
                    "id",
                    "uri",
                    "track_href",
                    "analysis_url",
                    "type",
                    "time_signature",
                ]
            )
            genre = model.predict(df)[0]

            # Get the color and emoji for the genre
            genre_style = genre_styles.get(
                genre.lower(), {"color": "#000000", "emoji": "🎵"}
            )
            genre_color = genre_style["color"]
            genre_emoji = genre_style["emoji"]

            # Calculate the best contrasting text color
            text_color = get_contrasting_text_color(genre_color)

            # Replace loading placeholder with actual genre display
            genre_placeholder.empty()
            genre_placeholder.markdown(
                f"""
                <div style='width: 100%; display: flex; justify-content: center;'>
                    <div style='background-color: {genre_color}; 
                                padding: 15px; border-radius: 8px; 
                                text-align: center; width: 50%;'>
                        <span style='color: {text_color}; font-size: 20px;'>
                            {genre_emoji} The predicted genre is: {genre} {genre_emoji}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
