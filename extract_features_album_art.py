import os
import pandas as pd
import numpy as np
import librosa
import cv2
import tensorflow as tf
from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

# Set the base path for your album data
BASE_PATH = "C:\\Album Art Generator\\album_data"  # Adjust this path
OUTPUT_CSV = "combined_features.csv"

# Function to extract audio features
def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    return tempo, mfcc_mean

# Function to extract image features
def extract_image_features(image_file):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Function to gather audio features into a DataFrame
def gather_audio_features():
    audio_features = []
    
    # Walk through the directories in the base path
    for album_dir in os.listdir(BASE_PATH):
        album_path = os.path.join(BASE_PATH, album_dir)
        if os.path.isdir(album_path):
            for filename in os.listdir(album_path):
                if filename.endswith(".wav") and "preview" in filename:
                    album_name = album_dir  # Using the directory name as album name
                    audio_file_path = os.path.join(album_path, filename)
                    tempo, mfcc_mean = extract_audio_features(audio_file_path)
                    audio_features.append([album_name, tempo] + mfcc_mean.tolist())
    
    columns = ['album_name', 'tempo'] + [f'mfcc_{i}' for i in range(1, 14)]
    audio_features_df = pd.DataFrame(audio_features, columns=columns)
    return audio_features_df

# Function to gather image features into a DataFrame
def gather_image_features():
    image_features = []
    
    # Walk through the directories in the base path
    for album_dir in os.listdir(BASE_PATH):
        album_path = os.path.join(BASE_PATH, album_dir)
        if os.path.isdir(album_path):
            album_art_path = os.path.join(album_path, "album_art.jpg")
            if os.path.exists(album_art_path):
                album_name = album_dir  # Using the directory name as album name
                features = extract_image_features(album_art_path)
                image_features.append([album_name] + features.tolist())
    
    columns = ['album_name'] + [f'feature_{i}' for i in range(1, 2049)]  # ResNet50 outputs 2048 features
    image_features_df = pd.DataFrame(image_features, columns=columns)
    return image_features_df

# Function to combine audio and image features
def combine_features(audio_features_df, image_features_df):
    # Print DataFrame columns for debugging
    print("Audio Features DataFrame Columns:", audio_features_df.columns)
    print("Image Features DataFrame Columns:", image_features_df.columns)
    
    # Check for the presence of the 'album_name' column
    if 'album_name' not in audio_features_df.columns:
        print("Error: 'album_name' column not found in audio_features_df")
    if 'album_name' not in image_features_df.columns:
        print("Error: 'album_name' column not found in image_features_df")
    
    # Merge the DataFrames
    combined_df = pd.merge(audio_features_df, image_features_df, on='album_name', how='inner')
    
    return combined_df

# Main execution
if __name__ == "__main__":
    audio_features_df = gather_audio_features()
    image_features_df = gather_image_features()
    
    combined_features_df = combine_features(audio_features_df, image_features_df)
    
    # Save combined DataFrame to CSV
    combined_features_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Combined features saved to {OUTPUT_CSV}")
