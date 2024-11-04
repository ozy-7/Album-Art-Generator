import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def preprocess_album_data(dataset_dir):
    """Preprocesses audio data in the dataset directory."""
    
    # Define the base directory for processed files
    processed_dir = os.path.join("C:\Album Art Generator", "processed_audio")
    
    # Loop through each album in the dataset
    for album_name in os.listdir(dataset_dir):
        album_path = os.path.join(dataset_dir, album_name)
        
        if os.path.isdir(album_path):
            # Create corresponding directory in the processed path
            processed_album_path = os.path.join(processed_dir, album_name)
            os.makedirs(processed_album_path, exist_ok=True)
            
            # Process each audio file in the album directory
            for audio_file in os.listdir(album_path):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(album_path, audio_file)
                    audio, sr = librosa.load(audio_path, sr=None)
                    
                    # Convert audio to mel spectrogram
                    mel_spectrogram = convert_to_mel_spectrogram(audio, sr, n_mels=256)  # Increased n_mels
                    
                    # Save the mel spectrogram as an image in the new path
                    output_filename = f"{audio_file.replace('.wav', '')}_mel.png"
                    output_path = os.path.join(processed_album_path, output_filename)
                    
                    save_spectrogram_image(mel_spectrogram, output_path)
                    print(f"Processed spectrogram saved to: {output_path}")

def convert_to_mel_spectrogram(audio, sr, n_mels=256):  # Default n_mels increased to 256
    """Converts audio to a mel spectrogram."""
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

def save_spectrogram_image(spectrogram, output_path):
    """Saves the mel spectrogram as an image file."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=22050, x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    
    # Set a higher DPI for better image resolution
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)  # Increased DPI to 300
    plt.close()

# Run the preprocessing function
dataset_dir = r"C:\Album Art Generator\album_data"
preprocess_album_data(dataset_dir)
