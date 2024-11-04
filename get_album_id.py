import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

# Set up Spotify credentials
SPOTIPY_CLIENT_ID = 'f4766aaeb32246d29dd57174b4756c06'
SPOTIPY_CLIENT_SECRET = 'f3ab65b143f645168092ed3cb25bab12'

client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to get popular albums based on playlists for a genre
def get_popular_albums_from_playlists(genre, limit=50):
    # Search for popular playlists related to the genre
    results = sp.search(q=f'{genre}', type='playlist', limit=5, market='US')
    playlists = results['playlists']['items']
    
    album_ids = []
    
    # Iterate through the playlists and gather albums from tracks
    for playlist in playlists:
        playlist_id = playlist['id']
        playlist_tracks = sp.playlist_tracks(playlist_id, limit=limit)['items']
        
        for item in playlist_tracks:
            track = item['track']
            album = track['album']
            album_id = album['id']
            album_name = album['name']
            artist_name = album['artists'][0]['name']
            
            # Append album details
            if (album_id, album_name, artist_name) not in album_ids:
                album_ids.append((album_id, album_name, artist_name))
    
    return album_ids

# Save album IDs to txt file containing only the IDs
def save_album_ids_to_file(album_ids, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for album_id, _, _ in album_ids:  # Only get the album ID
            f.write(f"{album_id}\n")
    print(f"Saved {len(album_ids)} album IDs to {filename}")

# Genres to search for
genres = ['rock', 'metal', 'pop']

# Fetch and save popular album IDs for each genre
all_album_ids = []
for genre in genres:
    album_ids = get_popular_albums_from_playlists(genre, limit=50)  # Adjust limit as needed
    all_album_ids.extend(album_ids)
    save_album_ids_to_file(album_ids, f"popular_{genre}_albums.txt")

# Optionally, save all album IDs together in one file
save_album_ids_to_file(all_album_ids, 'all_genres_popular_albums.txt')
