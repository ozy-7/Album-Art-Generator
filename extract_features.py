import os
import librosa
import numpy as np
import pandas as pd

def extract_features(audio_path):
    """Extracts audio features from the given audio file."""
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Feature extraction
    features = {}
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    features['tempo'] = tempo
    
    # Pitch (using piptrack)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    features['average_pitch'] = np.mean(pitches)

    # Rhythm (onset strength)
    onset_strength = librosa.onset.onset_strength(y=audio, sr=sr)
    features['average_onset_strength'] = np.mean(onset_strength)

    # Spectral Features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features['spectral_centroid'] = np.mean(spectral_centroid)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
    features['spectral_rolloff'] = np.mean(spectral_rolloff)

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
    features['zero_crossing_rate'] = np.mean(zero_crossing_rate)

    return features

def preprocess_album_data(dataset_dir):
    """Preprocesses audio data in the dataset directory."""
    
    # Initialize a list to collect features
    all_features = []
    
    # Loop through each album in the dataset
    for album_name in os.listdir(dataset_dir):
        album_path = os.path.join(dataset_dir, album_name)
        
        if os.path.isdir(album_path):
            for audio_file in os.listdir(album_path):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(album_path, audio_file)
                    
                    # Extract features from audio
                    features = extract_features(audio_path)
                    
                    # Add album and track names to features
                    features['album'] = album_name
                    features['track'] = audio_file
                    
                    # Append features to the list
                    all_features.append(features)
    
    # Convert the list of features to a DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Save DataFrame to CSV if needed
    features_df.to_csv(os.path.join(dataset_dir, "extracted_features.csv"), index=False)
    print("Features successfully extracted and saved to CSV.")

# Run the feature extraction
dataset_dir = r"C:\Album Art Generator\album_data"
preprocess_album_data(dataset_dir)