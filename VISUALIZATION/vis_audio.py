#!/usr/bin/env python3
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

def load_and_process_audio(audio_path, n_fft=1024, hop_length=512, n_mels=64):
    """Load and process audio file to create mel spectrogram."""
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Create mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        normalized=True
    )
    
    # Convert to mel spectrogram
    mel_spec = mel_transform(waveform)
    
    # Convert to decibel scale
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
        stype='power',
        top_db=80.0
    )
    mel_spec_db = amplitude_to_db(mel_spec)
    
    return mel_spec_db, sample_rate

def plot_multiple_spectrograms(spectrograms, mic_indices, title="Multi-Microphone Mel Spectrograms", save_path=None):
    """Plot multiple spectrograms in a grid layout."""
    n_mics = len(spectrograms)
    fig = plt.figure(figsize=(15, 3*n_mics))
    
    # Find global min and max for consistent colorbar
    vmin = min(spec.squeeze().numpy().min() for spec in spectrograms)
    vmax = max(spec.squeeze().numpy().max() for spec in spectrograms)
    
    for i, (spec, mic_idx) in enumerate(zip(spectrograms, mic_indices)):
        plt.subplot(n_mics, 1, i+1)
        # Squeeze out the channel dimension and transpose for correct orientation
        spec_np = spec.squeeze().numpy()
        plt.imshow(spec_np, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Microphone {mic_idx}')
        plt.ylabel('Mel Frequency')
        if i == n_mics-1:  # Only show x-label for bottom plot
            plt.xlabel('Time')
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_multiple_mics(base_path, trial_name, output_dir=None):
    """Visualize spectrograms from multiple microphones for a trial."""
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all microphone files for the trial
    mic_files = []
    mic_indices = []
    for idx in range(12, 17):  # idx12 to idx16
        pattern = f"{base_path}/{trial_name}/{trial_name}_mic*_idx{idx}.wav"
        files = glob.glob(pattern)
        if files:
            mic_files.append(files[0])
            mic_indices.append(idx)
    
    if not mic_files:
        print(f"No microphone files found for trial {trial_name}")
        return
    
    # Process all microphone files
    spectrograms = []
    for mic_file in mic_files:
        mel_spec_db, _ = load_and_process_audio(mic_file)
        spectrograms.append(mel_spec_db)
    
    # Generate output path if saving
    if output_dir:
        save_path = os.path.join(output_dir, f"{trial_name}_multi_mic_spectrograms.png")
    else:
        save_path = None
    
    # Plot all spectrograms
    plot_multiple_spectrograms(
        spectrograms, 
        mic_indices,
        title=f"Multi-Microphone Mel Spectrograms - {trial_name}",
        save_path=save_path
    )

def main():
    # Example usage
    base_path = "/media/frida/Extreme SSD/sounding_hand/yuemin/scissors/mic_raw"
    trial_name = "t1"  # Change this to the trial you want to visualize
    output_dir = "spectrogram_visualizations"
    
    visualize_multiple_mics(base_path, trial_name, output_dir)

if __name__ == "__main__":
    main()
