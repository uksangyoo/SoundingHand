import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA"

dataset_name = "0414_spam_pla"
trial_num = 1

output_base_dir = os.path.join(DATA_ROOT, dataset_name, "output")
os.makedirs(output_base_dir, exist_ok=True)

#-- Load Audio Files ---
audio_files_paths = [
    os.path.join(DATA_ROOT, dataset_name, dataset_name + "_audio", f"t{trial_num}_mic{i}_idx{13+i}.wav")
    for i in range(5)
]

# Create figure for spectrograms - vertically stacked
plt.figure(figsize=(12, 15))  # Tall figure for vertical stacking

# Process each audio file and create spectrograms
for idx, audio_path in enumerate(audio_files_paths, 1):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Calculate duration of 1 second in samples
    one_sec_samples = sr
    
    # Extract first 1 second for background noise profile
    if len(y) > one_sec_samples:
        bg_noise = y[:one_sec_samples]
        
        # Compute mel spectrogram of background noise
        bg_mel_spect = librosa.feature.melspectrogram(y=bg_noise, sr=sr, n_mels=128)
        bg_mel_spect_db = librosa.power_to_db(bg_mel_spect, ref=np.max)
        
        # Average the background noise profile across time
        bg_profile = np.mean(bg_mel_spect, axis=1, keepdims=True)
        
        # Compute mel spectrogram of the full signal
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        
        # Subtract background noise profile
        mel_spect_denoised = mel_spect - bg_profile
        
        # Ensure no negative values after subtraction
        mel_spect_denoised = np.maximum(mel_spect_denoised, 1e-10)
        
        # Convert to dB scale
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        mel_spect_denoised_db = librosa.power_to_db(mel_spect_denoised, ref=np.max)
        
        # Create two subplots - Original and Denoised
        plt.subplot(5, 2, 2*idx-1)
        librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mic {idx} - Original')
        
        plt.subplot(5, 2, 2*idx)
        librosa.display.specshow(mel_spect_denoised_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mic {idx} - Background Subtracted')
    else:
        # Handle case where audio is shorter than 1 second
        print(f"Warning: Audio file {audio_path} is shorter than 1 second. Skipping background subtraction.")
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        
        plt.subplot(5, 1, idx)
        librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Microphone {idx}')

# Adjust layout and save
plt.tight_layout()
plt.savefig(os.path.join(output_base_dir, f't{trial_num}_spectrograms_denoised.png'), dpi=300, bbox_inches='tight')
plt.close()

# Also save a separate visualization showing background noise profile for each microphone
plt.figure(figsize=(15, 8))
for idx, audio_path in enumerate(audio_files_paths, 1):
    y, sr = librosa.load(audio_path, sr=None)
    if len(y) > sr:  # Ensure audio is longer than 1 second
        bg_noise = y[:sr]
        bg_mel_spect = librosa.feature.melspectrogram(y=bg_noise, sr=sr, n_mels=128)
        bg_mel_spect_db = librosa.power_to_db(bg_mel_spect, ref=np.max)
        
        plt.subplot(1, 5, idx)
        librosa.display.specshow(bg_mel_spect_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mic {idx} - Background Noise (1s)')

plt.tight_layout()
plt.savefig(os.path.join(output_base_dir, f't{trial_num}_background_noise.png'), dpi=300, bbox_inches='tight')
plt.close()