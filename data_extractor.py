import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from pydub import AudioSegment
import requests
import re

# Set up Spotify credentials
SPOTIPY_CLIENT_ID = 'f4766aaeb32246d29dd57174b4756c06'
SPOTIPY_CLIENT_SECRET = 'f3ab65b143f645168092ed3cb25bab12'

client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Directory to save audio previews and album art
save_dir = "album_data"
os.makedirs(save_dir, exist_ok=True)

def sanitize_filename(filename):
    # Remove invalid characters for file names
    return re.sub(r'[<>:"/\\|?*]', '', filename)

def get_album_tracks(album_id):
    album_tracks = sp.album_tracks(album_id)
    tracks = album_tracks['items']
    
    track_info = []
    for track in tracks:
        track_id = track['id']
        track_name = sanitize_filename(track['name'])  # Sanitize track name
        preview_url = track.get('preview_url')  # Use .get() to avoid KeyError
        if preview_url:  # Only include tracks with a preview URL
            track_info.append((track_id, track_name, preview_url))
        
    return track_info

def get_album_art(album_id):
    album_data = sp.album(album_id)
    album_art_url = album_data['images'][0]['url']  # High-res album art
    return album_art_url

def download_audio_preview(album_dir, track_name, preview_url):
    if preview_url:
        # Download the 30-second preview audio
        response = requests.get(preview_url)
        audio_path = os.path.join(album_dir, f"{track_name}.mp3")  # Save to the album directory
        
        # Save the 30-second preview to file
        with open(audio_path, "wb") as f:
            f.write(response.content)

        # Load the audio (which is already 30 seconds)
        song = AudioSegment.from_mp3(audio_path)

        # Save the 30-second preview directly as WAV
        segment_path = os.path.join(album_dir, f"{track_name}_preview_30s.wav")  # Save to the album directory
        song.export(segment_path, format="wav")
        print(f"Saved 30-second segment for {track_name}")
        
        # Delete the 30-second preview
        os.remove(audio_path)
        print(f"Deleted 30-second preview for {track_name}")
    else:
        print(f"No preview available for {track_name}")

def download_album_art(album_art_url, album_dir):
    response = requests.get(album_art_url)
    album_art_path = os.path.join(album_dir, "album_art.jpg")  # Save to the album directory
    
    with open(album_art_path, "wb") as f:
        f.write(response.content)
    print("Saved album art")

# Read album IDs from the text file
with open('C:\\Album Art Generator\\all_genres_popular_albums.txt', 'r') as f:
    album_ids = [line.strip() for line in f.readlines()]

# Extract 30-second segments and album art for each album
for album_id in album_ids:
    # Get tracks from the album
    tracks = get_album_tracks(album_id)

    # Only proceed if there are tracks to process
    if tracks:
        # Get album name for directory and sanitize it
        album_name = sanitize_filename(sp.album(album_id)['name'])  # Sanitize album name
        album_dir = os.path.join(save_dir, album_name)
        os.makedirs(album_dir, exist_ok=True)

        for track_id, track_name, preview_url in tracks:
            download_audio_preview(album_dir, track_name, preview_url)

        # Get and save album art
        album_art_url = get_album_art(album_id)
        download_album_art(album_art_url, album_dir)
    else:
        print(f"No valid tracks for album ID: {album_id}")
