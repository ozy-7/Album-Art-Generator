import os
import re
import pandas as pd
import requests
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from bs4 import BeautifulSoup
import librosa
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import unicodedata
import logging
from textblob import TextBlob
import streamlit as st
from main_v3_last import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, GENIUS_API_KEY, HUGGINGFACE_TOKEN


# ---------------------------- CONFIGURATIONS ---------------------------- #

# Fetch API credentials from environment variables
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
GENIUS_API_KEY = os.getenv("GENIUS_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Path to your dataset with genres
DATASET_PATH = "final_dataset_with_genres.csv"

# Output directory for generated album art
OUTPUT_DIR = "generated_album_art"

# Stable Diffusion Model ID (change if using a different model)
STABLE_DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# ---------------------------- LOGGING CONFIGURATION ---------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

# ---------------------------- FUNCTIONS ---------------------------- #

def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    """
    try:
        df = pd.read_csv(dataset_path)
        logging.info(f"Dataset loaded successfully with {len(df)} records.")
        return df
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return None

def authenticate_spotify(client_id: str, client_secret: str) -> Spotify:
    """
    Authenticate with Spotify and return a Spotipy client.
    """
    try:
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = Spotify(auth_manager=auth_manager)
        logging.info("Authenticated with Spotify successfully.")
        return sp
    except Exception as e:
        logging.error(f"Spotify authentication failed: {e}")
        return None

def fetch_genres_spotify(sp: Spotify, artist_name: str, track_name: str) -> list:
    """
    Fetch genres associated with the given artist from Spotify.
    """
    try:
        results = sp.search(q=f'artist:{artist_name} track:{track_name}', type='track', limit=1)
        tracks = results['tracks']['items']
        if not tracks:
            logging.warning(f"Track '{track_name}' by '{artist_name}' not found on Spotify.")
            return []
        track = tracks[0]
        artist_id = track['artists'][0]['id']
        artist_info = sp.artist(artist_id)
        genres = artist_info['genres']  # List of genre strings
        logging.info(f"Found genres for '{artist_name}': {genres}")
        return genres
    except Exception as e:
        logging.error(f"Error fetching genres for '{artist_name}': {e}")
        return []

def fetch_album_spotify(sp: Spotify, artist_name: str, track_name: str) -> str:
    """
    Fetch the album name for the given artist and track from Spotify.
    """
    try:
        results = sp.search(q=f'artist:{artist_name} track:{track_name}', type='track', limit=1)
        tracks = results['tracks']['items']
        if not tracks:
            logging.warning(f"Track '{track_name}' by '{artist_name}' not found on Spotify.")
            return "Unknown Album"
        track = tracks[0]
        album_name = track['album']['name']
        logging.info(f"Found album '{album_name}' for '{track_name}' by '{artist_name}'.")
        return album_name
    except Exception as e:
        logging.error(f"Error fetching album for '{artist_name}': {e}")
        return "Unknown Album"

def fetch_lyrics_from_genius(song_title: str, artist_name: str, genius_api_key: str) -> str:
    """
    Fetch lyrics by searching song_title + artist_name on Genius,
    then scraping from the first search result.
    """
    GENIUS_BASE_URL = "https://api.genius.com"
    search_url = f"{GENIUS_BASE_URL}/search"
    headers = {"Authorization": f"Bearer {genius_api_key}"}
    params = {"q": f"{song_title} {artist_name}"}

    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()

        hits = response.json().get("response", {}).get("hits", [])
        if not hits:
            logging.info(f"No lyrics found for '{song_title}' by '{artist_name}'.")
            return ""

        # Take the first hit
        song_url = hits[0]["result"]["url"]
        lyrics_response = requests.get(song_url)
        lyrics_response.raise_for_status()

        soup = BeautifulSoup(lyrics_response.text, "html.parser")
        lyrics_divs = soup.find_all("div", {"data-lyrics-container": "true"})
        lyrics = "\n".join([div.get_text(separator="\n", strip=True) for div in lyrics_divs]).strip()

        # Remove bracketed text like [Chorus], [Verse], etc.
        clean_lyrics = re.sub(r"\[.*?\]", "", lyrics).strip()
        # Normalize the unicode
        clean_lyrics = unicodedata.normalize("NFKD", clean_lyrics)

        if clean_lyrics:
            logging.info(f"Lyrics found and fetched for '{song_title}' by '{artist_name}'.")
            return clean_lyrics
        else:
            logging.info(f"Lyrics not found for '{song_title}' by '{artist_name}'.")
            return ""

    except Exception as e:
        logging.error(f"Error fetching lyrics for '{song_title}' by '{artist_name}': {e}")
        return ""

def extract_audio_features(audio_path: str) -> np.ndarray:
    """
    Extract audio features using Librosa.
    Returns a feature vector as a NumPy array.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # Extract Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # Extract Spectral Contrast
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_contrast_mean = np.mean(spec_contrast, axis=1)
        spec_contrast_std = np.std(spec_contrast, axis=1)

        # Extract Tonnetz
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        tonnetz_std = np.std(tonnetz, axis=1)

        # Aggregate all features into a single array
        features = np.concatenate([
            mfccs_mean,  # mfcc_1_mean to mfcc_20_mean
            mfccs_std,   # mfcc_1_std to mfcc_20_std
            chroma_mean, # chroma_1_mean to chroma_N_mean
            chroma_std,  # chroma_1_std to chroma_N_std
            spec_contrast_mean,  # spectral_contrast_1_mean to spectral_contrast_N_mean
            spec_contrast_std,   # spectral_contrast_1_std to spectral_contrast_N_std
            tonnetz_mean,        # tonnetz_1_mean to tonnetz_N_mean
            tonnetz_std          # tonnetz_1_std to tonnetz_N_std
        ])

        logging.info(f"Audio features extracted successfully from '{audio_path}'.")
        return features

    except Exception as e:
        logging.error(f"Error extracting audio features from '{audio_path}': {e}")
        return None

def compute_feature_statistics(df: pd.DataFrame) -> dict:
    """
    Compute percentile-based thresholds for selected audio features.
    Returns a dictionary with thresholds.
    """
    feature_stats = {}

    try:
        # Compute 25th, 50th, and 75th percentiles for means of each feature type

        # MFCC Means
        mfcc_mean_columns = [col for col in df.columns if re.match(r'mfcc_\d+_mean', col)]
        mfcc_mean_values = df[mfcc_mean_columns].values.flatten()
        feature_stats['mfcc_mean_25'] = np.percentile(mfcc_mean_values, 25)
        feature_stats['mfcc_mean_50'] = np.percentile(mfcc_mean_values, 50)
        feature_stats['mfcc_mean_75'] = np.percentile(mfcc_mean_values, 75)

        # Chroma Means
        chroma_mean_columns = [col for col in df.columns if re.match(r'chroma_\d+_mean', col)]
        chroma_mean_values = df[chroma_mean_columns].values.flatten()
        feature_stats['chroma_mean_25'] = np.percentile(chroma_mean_values, 25)
        feature_stats['chroma_mean_50'] = np.percentile(chroma_mean_values, 50)
        feature_stats['chroma_mean_75'] = np.percentile(chroma_mean_values, 75)

        # Spectral Contrast Means
        spec_contrast_mean_columns = [col for col in df.columns if re.match(r'spectral_contrast_\d+_mean', col)]
        spec_contrast_mean_values = df[spec_contrast_mean_columns].values.flatten()
        feature_stats['spectral_contrast_mean_25'] = np.percentile(spec_contrast_mean_values, 25)
        feature_stats['spectral_contrast_mean_50'] = np.percentile(spec_contrast_mean_values, 50)
        feature_stats['spectral_contrast_mean_75'] = np.percentile(spec_contrast_mean_values, 75)

        # Tonnetz Means
        tonnetz_mean_columns = [col for col in df.columns if re.match(r'tonnetz_\d+_mean', col)]
        tonnetz_mean_values = df[tonnetz_mean_columns].values.flatten()
        feature_stats['tonnetz_mean_25'] = np.percentile(tonnetz_mean_values, 25)
        feature_stats['tonnetz_mean_50'] = np.percentile(tonnetz_mean_values, 50)
        feature_stats['tonnetz_mean_75'] = np.percentile(tonnetz_mean_values, 75)

        logging.info("Computed feature percentile statistics from the dataset.")
        return feature_stats
    except Exception as e:
        logging.error(f"Error computing feature statistics: {e}")
        return feature_stats

def interpret_audio_features(features: np.ndarray, feature_stats: dict) -> list:
    """
    Interpret numerical audio features into descriptive textual attributes
    based on dataset feature statistics using percentile thresholds.
    Returns a list of descriptors.
    """
    descriptors = []

    try:
        # Timbre Complexity based on MFCC Means (indices 0-19)
        mfcc_mean = np.mean(features[0:20])
        if mfcc_mean > feature_stats.get('mfcc_mean_75', 0):
            descriptors.append("very complex timbre")
        elif mfcc_mean > feature_stats.get('mfcc_mean_50', 0):
            descriptors.append("complex timbre")
        elif mfcc_mean > feature_stats.get('mfcc_mean_25', 0):
            descriptors.append("moderate timbre")
        elif mfcc_mean < feature_stats.get('mfcc_mean_25', 0):
            descriptors.append("simple timbre")
        else:
            descriptors.append("neutral timbre")

        # Harmony Richness based on Chroma Means (indices 40-59)
        chroma_mean = np.mean(features[40:60])
        if chroma_mean > feature_stats.get('chroma_mean_75', 0):
            descriptors.append("very rich harmony")
        elif chroma_mean > feature_stats.get('chroma_mean_50', 0):
            descriptors.append("rich harmony")
        elif chroma_mean > feature_stats.get('chroma_mean_25', 0):
            descriptors.append("balanced harmony")
        elif chroma_mean < feature_stats.get('chroma_mean_25', 0):
            descriptors.append("minimal harmony")
        else:
            descriptors.append("neutral harmony")

        # Brightness based on Spectral Contrast Means (indices 80-99)
        spec_contrast_mean = np.mean(features[80:100])
        if spec_contrast_mean > feature_stats.get('spectral_contrast_mean_75', 0):
            descriptors.append("very bright tones")
        elif spec_contrast_mean > feature_stats.get('spectral_contrast_mean_50', 0):
            descriptors.append("bright tones")
        elif spec_contrast_mean > feature_stats.get('spectral_contrast_mean_25', 0):
            descriptors.append("neutral brightness")
        elif spec_contrast_mean < feature_stats.get('spectral_contrast_mean_25', 0):
            descriptors.append("very dark tones")
        else:
            descriptors.append("dark tones")

        # Tonal Stability based on Tonnetz Means (indices 120-139)
        tonnetz_mean = np.mean(features[120:140])
        if tonnetz_mean > feature_stats.get('tonnetz_mean_75', 0):
            descriptors.append("very stable tonal structure")
        elif tonnetz_mean > feature_stats.get('tonnetz_mean_50', 0):
            descriptors.append("stable tonal structure")
        elif tonnetz_mean > feature_stats.get('tonnetz_mean_25', 0):
            descriptors.append("balanced tonal structure")
        elif tonnetz_mean < feature_stats.get('tonnetz_mean_25', 0):
            descriptors.append("unstable tonal structure")
        else:
            descriptors.append("neutral tonal structure")

        logging.info(f"Interpreted audio features into descriptors: {descriptors}")
        return descriptors

    except Exception as e:
        logging.error(f"Error interpreting audio features: {e}")
        return descriptors

def genres_to_style(genres: list) -> str:
    """
    Convert a list of Spotify genres into a descriptive style for Stable Diffusion.
    Handles multiple genres by combining their styles.
    """
    style_text = []

    # Define genre to style mapping
    genre_style_mapping = {
        "metal": "dark and heavy style (metal)",
        "metalcore": "aggressive and intense style (metalcore)",
        "hard rock": "bold and edgy style (hard rock)",
        "rock": "loud and driving style (rock)",
        "pop": "energetic and bright style (pop)",
        "hip hop": "urban hip-hop style",
        "rap": "urban hip-hop style",
        "electronic": "electronic dance style",
        "dance": "electronic dance style",
        "jazz": "smooth and jazzy style",
        "classical": "dramatic and orchestral style",
        "r&b": "soulful R&B style",
        "reggae": "laid-back reggae style",
        "blues": "moody blues style",
        "folk": "folksy acoustic style",
        "country": "country-western style",
        "instrumental": "ambient instrumental style",
        "punk": "rebellious punk style",
        "soul": "deep soulful style",
        "funk": "groovy funk style",
        "gospel": "uplifting gospel style",
        # More can be added...
    }

    # Iterate through genres and map
    for genre in genres:
        mapped = False
        for key, style in genre_style_mapping.items():
            if key in genre.lower():
                style_text.append(style)
                logging.info(f"Mapped genre '{genre}' to style '{style}'.")
                mapped = True
                break  # Take the first matching genre
        if not mapped:
            logging.info(f"No mapping found for genre '{genre}'. Using default 'mysterious style'.")
            style_text.append("mysterious style")

    # Combine styles, removing duplicates
    unique_styles = list(dict.fromkeys(style_text))
    combined_style = ", ".join(unique_styles)

    return combined_style

def analyze_lyrics_emotion(lyrics: str) -> list:
    """
    Analyze lyrics to determine the emotional tone or themes.
    Returns a list of emotion-based descriptors.
    """
    emotions = []

    try:
        blob = TextBlob(lyrics)
        sentiment = blob.sentiment.polarity  # -1 to 1

        if sentiment > 0.5:
            emotions.append("uplifting")
        elif sentiment > 0:
            emotions.append("positive")
        elif sentiment < -0.5:
            emotions.append("somber")
        elif sentiment < 0:
            emotions.append("negative")
        else:
            emotions.append("neutral")

        logging.info(f"Analyzed lyrics sentiment: {sentiment}, emotions: {emotions}")
        return emotions

    except Exception as e:
        logging.error(f"Error analyzing lyrics sentiment: {e}")
        return emotions

def build_prompt(style_text: str, audio_descriptors: list, lyrics: str, emotions: list, max_len=150) -> str:
    """
    Combine style, audio descriptors, emotions, and truncated lyrics into a single text prompt.
    """
    if not lyrics:
        lyrics = "no lyrics found"

    # Remove newlines and truncate if necessary
    short_lyrics = lyrics.replace("\n", " ")
    if len(short_lyrics) > max_len:
        short_lyrics = short_lyrics[:max_len] + "..."

    # Combine audio descriptors into a descriptive phrase
    audio_description = ", ".join(audio_descriptors)

    # Combine emotion descriptors
    emotion_description = ", ".join(emotions)

    # Add visual elements based on descriptors
    visual_elements = ""
    if "bright tones" in audio_descriptors:
        visual_elements += ", featuring vibrant colors"
    if "dark tones" in audio_descriptors:
        visual_elements += ", incorporating dark hues"
    if "complex timbre" in audio_descriptors:
        visual_elements += ", with intricate patterns"
    if "simple timbre" in audio_descriptors:
        visual_elements += ", showcasing minimalist design"

    if emotions:
        visual_elements += f", evoking a {emotion_description} atmosphere"

    prompt = f"An album cover in a {style_text}, characterized by {audio_description}{visual_elements}, inspired by lyrics: '{short_lyrics}'"
    logging.info(f"Built prompt: {prompt}")
    return prompt

def setup_stable_diffusion(model_id: str, huggingface_token: str = None) -> StableDiffusionPipeline:
    """
    Set up the Stable Diffusion pipeline.
    """
    try:
        if huggingface_token:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                use_auth_token=huggingface_token
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_id)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Stable Diffusion pipeline initialized successfully.")
        return pipe
    except Exception as e:
        logging.error(f"Failed to initialize Stable Diffusion pipeline: {e}")
        return None

def generate_album_art(prompt: str,
                       pipe: StableDiffusionPipeline,
                       num_inference_steps: int = 50,
                       guidance_scale: float = 7.5) -> Image.Image:
    """
    Run the Stable Diffusion pipeline with the text prompt.
    """
    logging.info(f"Generating album art with prompt: {prompt}")
    try:
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            result = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
        logging.info("Album art generated successfully.")
        return result.images[0]
    except Exception as e:
        logging.error(f"Image generation failed: {e}")
        return None

def sanitize_filename(name: str) -> str:
    """
    Sanitize the filename by removing or replacing illegal characters.
    """
    return re.sub(r'[\\/*?:"<>|]',"", name)

# ---------------------------- MAIN WORKFLOW ---------------------------- #

def main():
    st.title("🎨 Dynamic Album Art Generator")
    st.write("Generate custom album art based on your favorite song's audio features.")

    # Sidebar for API credentials (optional)
    with st.sidebar:
        st.header("API Credentials")
        spotify_id = st.text_input("Spotify Client ID", value=SPOTIFY_CLIENT_ID or "")
        spotify_secret = st.text_input("Spotify Client Secret", value=SPOTIFY_CLIENT_SECRET or "", type="password")
        genius_key = st.text_input("Genius API Key", value=GENIUS_API_KEY or "")
        huggingface_token = st.text_input("Huggingface Token (optional)", value=HUGGINGFACE_TOKEN or "", type="password")

        # Optionally, add a button to update environment variables or handle securely
        # For simplicity, we'll proceed assuming they are set

    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "aac", "ogg"])

    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with open("temp_audio_file", "wb") as f:
            f.write(uploaded_file.getbuffer())
        audio_path = "temp_audio_file"

        # Display filename
        filename = uploaded_file.name
        st.write(f"Uploaded File: **{filename}**")

        # Parse artist and song name from filename
        match = re.match(r'(?P<artist>.+?)\s*-\s*(?P<song>.+?)\.(wav|mp3|flac|aac|ogg)$', filename, re.IGNORECASE)
        if not match:
            st.warning("Filename format incorrect. Expected format: 'Artist - Song.ext'. Please enter details manually.")
            artist_name = st.text_input("Artist Name", value="Unknown Artist")
            song_title = st.text_input("Song Title", value="Unknown Song")
        else:
            artist_name = match.group('artist').strip()
            song_title = match.group('song').strip()
            st.write(f"Parsed Artist: **{artist_name}**, Song: **{song_title}**")

        # Authenticate with Spotify
        if spotify_id and spotify_secret:
            sp = authenticate_spotify(spotify_id, spotify_secret)
            if not sp:
                st.error("Spotify Authentication Failed. Please check your credentials.")
                return
        else:
            st.error("Spotify Client ID and Secret are required. Please enter them in the sidebar.")
            return

        # Load dataset
        if os.path.exists(DATASET_PATH):
            df = load_dataset(DATASET_PATH)
            if df is None:
                st.error("Failed to load dataset. Please check the CSV file.")
                return
        else:
            st.error(f"Dataset file '{DATASET_PATH}' not found.")
            return

        # Compute feature statistics
        feature_stats = compute_feature_statistics(df)
        if not feature_stats:
            st.error("Failed to compute feature statistics.")
            return

        # Fetch album name
        album_name = fetch_album_spotify(sp, artist_name, song_title)
        if album_name != "Unknown Album":
            st.write(f"Album Name: **{album_name}**")
        else:
            st.write("Album Name: **Unknown Album**")

        # Fetch genres
        genres = fetch_genres_spotify(sp, artist_name, song_title)
        if genres:
            st.write(f"Genres: **{', '.join(genres)}**")
        else:
            st.write("Genres: **Unknown**")

        # Attempt to fetch lyrics
        if genius_key:
            lyrics = fetch_lyrics_from_genius(song_title, artist_name, genius_key)
            if lyrics:
                st.success("Lyrics found and fetched.")
            else:
                st.warning("Lyrics not found.")
        else:
            st.warning("Genius API Key not provided. Lyrics fetching skipped.")
            lyrics = ""

        # Extract audio features
        features = extract_audio_features(audio_path)
        if features is not None:
            # Interpret audio features into descriptors using dataset feature statistics
            audio_descriptors = interpret_audio_features(features, feature_stats)
            st.write(f"Audio Descriptors: **{', '.join(audio_descriptors)}**")
        else:
            # Assign default descriptors if features are unavailable
            audio_descriptors = ["moderate timbre", "balanced harmony", "neutral brightness", "balanced tonal structure"]
            st.warning("Audio feature extraction failed. Assigned default descriptors.")

        # Analyze lyrics for emotions
        emotions = analyze_lyrics_emotion(lyrics)
        if emotions:
            st.write(f"Lyrics Emotion: **{', '.join(emotions)}**")
        else:
            st.write("Lyrics Emotion: **Neutral**")

        # Map genres to styles
        style_text = genres_to_style(genres)

        # Build prompt
        prompt = build_prompt(style_text, audio_descriptors, lyrics, emotions)

        st.write(f"**Generated Prompt:** {prompt}")

        # Generate album art button
        if st.button("Generate Album Art"):
            with st.spinner("Generating album art..."):
                # Set up Stable Diffusion pipeline
                pipe = setup_stable_diffusion(STABLE_DIFFUSION_MODEL_ID, huggingface_token)
                if not pipe:
                    st.error("Failed to initialize Stable Diffusion pipeline.")
                    return

                # Generate album art
                image = generate_album_art(prompt, pipe)
                if image:
                    # Save image
                    if not os.path.exists(OUTPUT_DIR):
                        os.makedirs(OUTPUT_DIR)
                    sanitized_song_title = sanitize_filename(song_title)
                    sanitized_artist_name = sanitize_filename(artist_name)
                    output_filename = f"{sanitized_song_title} - {sanitized_artist_name}.png"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    try:
                        image.save(output_path)
                        st.success(f"Album art generated and saved at '{output_path}'.")
                        st.image(image, caption=f"Album Art for '{song_title}' by '{artist_name}'", use_column_width=True)
                    except Exception as e:
                        st.error(f"Failed to save image: {e}")
                else:
                    st.error("Failed to generate album art.")

        # Clean up temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

# ---------------------------- RUN ---------------------------- #

if __name__ == "__main__":
    main()