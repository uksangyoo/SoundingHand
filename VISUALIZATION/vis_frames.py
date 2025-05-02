import os
import re
import glob
import json
import h5py
import cv2
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile

def h5_to_video(h5_path, output_video_path, fps=30):
    """
    Convert RGB frames from H5 file to MP4 video.
    
    Args:
        h5_path (str): Path to the H5 file
        output_video_path (str): Path where the output video will be saved
        fps (int): Frames per second for the output video
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    with h5py.File(h5_path, 'r') as h5_file:
        # Get total number of frames
        total_frames = h5_file['rgb'].shape[0]
        print(f"Total frames in H5 file: {total_frames}")
        
        # Get frame dimensions from the first frame
        first_frame = h5_file['rgb'][0][...]
        height, width = first_frame.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Process each frame
        for frame_idx in range(total_frames):
            # Read frame from H5 file
            frame = h5_file['rgb'][frame_idx][...].copy()
            
            # Write frame to video
            video_writer.write(frame)
            
            # Print progress
            if frame_idx % 100 == 0:
                print(f"Processed frame {frame_idx}/{total_frames}")
    
    # Release video writer
    video_writer.release()
    print(f"Video saved to: {output_video_path}")

def get_audio_paths(base_folder, trial_name):
    pattern = f"{base_folder}/{trial_name}/{trial_name}_mic*_idx*.wav"
    wav_files = glob.glob(pattern)
    mic_map = {}
    for p in wav_files:
        m = re.search(r'mic(\d+)_idx', os.path.basename(p))
        if m:
            mic_map[int(m.group(1))] = p
    return mic_map

def get_video_info(json_path):
    with open(json_path) as f:
        d = json.load(f)
    cam0 = d['cameras']['cam0']
    n_cam0 = cam0['total_frames']
    cam1 = d['cameras']['cam1']
    n_cam1 = cam1['total_frames']
    offset = d['differences_ms']['cam1_minus_cam0']['diffs_ms'][0]
    fl = 1 / 30 * 1000
    if offset < -1.5 * fl:
        print("Case 1: cam0[0], cam1[2]")
        start_cam = cam0['timestamps_ms'][0] / 1000.0
        n_cam = min(n_cam0, n_cam1 - 2)
    elif offset < -0.5 * fl:
        print("Case 2: cam0[0], cam1[1]")
        start_cam = cam0['timestamps_ms'][0] / 1000.0
        n_cam = min(n_cam0, n_cam1 - 1)
    elif offset < 0.5 * fl:
        print("Case 3: cam0[0], cam1[0]")
        start_cam = cam0['timestamps_ms'][0] / 1000.0
        n_cam = min(n_cam0, n_cam1)
    elif offset < 1.5 * fl:
        print("Case 4: cam0[1], cam1[0]")
        start_cam = cam0['timestamps_ms'][1] / 1000.0
        n_cam = min(n_cam0 - 1, n_cam1)
    else:
        print("Case 5: cam0[2], cam1[0]")
        start_cam = cam0['timestamps_ms'][2] / 1000.0
        n_cam = min(n_cam0 - 2, n_cam1)
    n_cam -= 1
    return start_cam, n_cam

def get_mic0_start(json_path):
    with open(json_path) as f:
        d = json.load(f)
    for rec in d.get('recordings', []):
        if rec.get('mic_number') == 0:
            return rec['start_time_ms'] / 1000.0
    raise RuntimeError("mic0 entry not found")

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

def create_synchronized_visualization(h5_path, mic_paths, sync_log_root, trial_name, output_path):
    """
    Create a visualization showing video frames synchronized with audio spectrograms.
    """
    # Get synchronization info
    vid_start, num_frames = get_video_info(os.path.join(sync_log_root, f"{trial_name}_cam_sync.json"))
    audio0_start = get_mic0_start(os.path.join(sync_log_root, f"{trial_name}_mic_sync.json"))
    audio_offset = 6.5e-3
    frame_len = 1.0 / 30.0

    # Load video frames
    with h5py.File(h5_path, 'r') as h5_file:
        frames = []
        for i in range(num_frames):
            frame = h5_file['rgb'][i][...].copy()
            frames.append(frame)

    # Load and process audio for each microphone
    spectrograms = []
    mic_indices = sorted(mic_paths.keys())
    for mic in mic_indices:
        mel_spec_db, _ = load_and_process_audio(mic_paths[mic])
        spectrograms.append(mel_spec_db)

    # Create figure with subplots
    n_mics = len(spectrograms)
    fig = plt.figure(figsize=(15, 3*n_mics + 3))
    
    # Add video frame subplot at the top
    ax_video = plt.subplot(n_mics + 1, 1, 1)
    ax_video.axis('off')
    
    # Create spectrogram subplots
    ax_specs = []
    for i in range(n_mics):
        ax = plt.subplot(n_mics + 1, 1, i + 2)
        ax_specs.append(ax)
    
    # Find global min and max for consistent colorbar
    vmin = min(spec.squeeze().numpy().min() for spec in spectrograms)
    vmax = max(spec.squeeze().numpy().max() for spec in spectrograms)
    
    # Plot initial spectrograms
    lines = []
    for i, (spec, mic_idx) in enumerate(zip(spectrograms, mic_indices)):
        spec_np = spec.squeeze().numpy()
        im = ax_specs[i].imshow(spec_np, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        ax_specs[i].set_title(f'Microphone {mic_idx}')
        ax_specs[i].set_ylabel('Mel Frequency')
        if i == n_mics-1:
            ax_specs[i].set_xlabel('Time')
        plt.colorbar(im, ax=ax_specs[i], format='%+2.0f dB')
        # Add vertical line for current frame
        line = ax_specs[i].axvline(x=0, color='r', linestyle='--')
        lines.append(line)
    
    # Show initial video frame
    img = ax_video.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
    ax_video.set_title('Video Frame')

    def update(frame):
        # Update video frame
        img.set_data(cv2.cvtColor(frames[frame], cv2.COLOR_BGR2RGB))
        
        # Calculate time position for this frame
        time_pos = vid_start + frame * frame_len
        
        # Update vertical lines in spectrograms
        for i, mic in enumerate(mic_indices):
            # Calculate local start time for this mic
            ls = vid_start - audio0_start - 1.0/60.0 + mic * audio_offset
            # Convert time to spectrogram x-coordinate
            x_pos = (time_pos - ls) * 44100 / 512  # Convert time to spectrogram bins
            lines[i].set_xdata(x_pos)
        
        return [img] + lines

    # Create animation
    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000/30, blit=True)
    
    # Save animation
    anim.save(output_path, writer='ffmpeg', fps=30, dpi=100)
    plt.close()

def main():
    # Paths
    DATA_ROOT = "/media/frida/Extreme SSD/sounding_hand/yuemin"
    SAVE_ROOT = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA"
    
    # Process scissors t1
    object_directory = "scissors"
    trial_name = "t1"
    
    # Get paths
    mic_raw_root = os.path.join(DATA_ROOT, object_directory, "mic_raw")
    sync_log_root = os.path.join(DATA_ROOT, object_directory, "sync_log")
    h5_path = os.path.join(DATA_ROOT, object_directory, "h5", trial_name)
    
    # Get audio paths
    mic_paths = get_audio_paths(mic_raw_root, trial_name)
    if not mic_paths:
        print(f"No mic files found for {object_directory} {trial_name}!")
        return
    
    # Get H5 file path
    h5_files = sorted([f for f in os.listdir(h5_path) if f.endswith('.h5')])
    if not h5_files:
        print(f"No H5 files found for {object_directory} {trial_name}!")
        return
    
    # Create output directory
    output_dir = os.path.join(SAVE_ROOT, object_directory, trial_name, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization for camera 0
    output_path = os.path.join(output_dir, f"{trial_name}_cam0_sync.mp4")
    create_synchronized_visualization(
        os.path.join(h5_path, h5_files[0]),
        mic_paths,
        sync_log_root,
        trial_name,
        output_path
    )
    
    # Create visualization for camera 1
    output_path = os.path.join(output_dir, f"{trial_name}_cam1_sync.mp4")
    create_synchronized_visualization(
        os.path.join(h5_path, h5_files[1]),
        mic_paths,
        sync_log_root,
        trial_name,
        output_path
    )

if __name__ == "__main__":
    main()
