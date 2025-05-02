import os 
import sys 
import torch 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import librosa
import librosa.display # Added for specshow
import soundfile as sf
# from PIL import Image # No longer needed if not loading frame images
import re
import imageio # Added for video writing
import shutil # Added for removing temporary directory
import matplotlib.colors as mcolors # Added for color normalization


if __name__ == "__main__":
    DATA_ROOT = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA"
    VIDEO_FRAMERATE = 30
    AUDIO_CLIP_DURATION_SECONDS = 0.02

    # --- Spectrogram Parameters ---
    N_FFT = 512 
    HOP_LENGTH = 128 #
    N_MELS = 64 
    FMIN = 20 
    FMAX = 8000 
    SPEC_CMAP = 'magma'

    dataset_name = "0414_spam_pla"
    trial_num = 1
    
    output_base_dir = os.path.join(DATA_ROOT, dataset_name, "output")
    
    # --- Output Directories ---
    audio_output_directory = os.path.join(output_base_dir, "audio_clips") # Renamed for clarity
    spectrogram_image_temp_dir = os.path.join(output_base_dir, "temp_spectrogram_images")
    video_output_path = os.path.join(output_base_dir, f"{dataset_name}_t{trial_num}_spectrogram_video.mp4")

    os.makedirs(audio_output_directory, exist_ok=True)
    os.makedirs(spectrogram_image_temp_dir, exist_ok=True)
    print(f"Audio clips output directory: {audio_output_directory}")
    print(f"Temporary spectrogram images directory: {spectrogram_image_temp_dir}")
    print(f"Output video path: {video_output_path}")

    # --- Load Audio Files ---
    audio_files_paths = [
        os.path.join(DATA_ROOT, dataset_name, dataset_name + "_audio", f"t{trial_num}_mic{i}_idx{13+i}.wav")
        for i in range(5)
    ]
    
    audios = []
    sr = None
    min_audio_len_samples = float('inf') 

    print("Loading audio files...")
    for i, audio_file in enumerate(audio_files_paths):
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file not found: {audio_file}. Skipping.")
            audios.append(None) 
            continue
            
        audio, current_sr = librosa.load(audio_file, sr=None)
        audios.append(audio)
        if sr is None:
            sr = current_sr
        elif sr != current_sr:
            raise ValueError(f"Inconsistent sample rates detected! Expected {sr}, got {current_sr} for {audio_file}")
        
        min_audio_len_samples = min(min_audio_len_samples, len(audio))
        print(f"Loaded mic {i}: shape={audio.shape}, sr={current_sr}")

    if sr is None:
        raise ValueError("No audio files were successfully loaded.")

    print(f"Using sample rate: {sr}")
    print(f"Minimum audio length (samples): {min_audio_len_samples}")


    # --- Find and Sort Frame Files ---
    visualization_directory = os.path.join(DATA_ROOT, dataset_name, "output") # Use output_base_dir? Yes.
    visualization_directory = output_base_dir # Corrected path
    pattern = re.compile(r"visualization_(\d{5})\.png")
    frames = []
    if os.path.exists(visualization_directory):
        for fname in os.listdir(visualization_directory):
            match = pattern.match(fname)
            if match:
                frame_number = int(match.group(1))
                frames.append(frame_number) 
    else:
        print(f"Warning: Visualization directory not found: {visualization_directory}")

    frames.sort() 

    if not frames:
        print("No visualization frames found matching the pattern.")
        sys.exit()
        
    print(f"Found {len(frames)} frames. Processing audio clips and calculating spectrograms...")

    # --- Process Each Frame (Pass 1: Audio Clips & Spectrogram Calculation) ---
    half_window_samples = int(AUDIO_CLIP_DURATION_SECONDS / 2 * sr)
    total_window_samples = half_window_samples * 2 

    processed_data = {} # Store data for the second pass {frame_num: {'audio': multi_channel_audio, 'specs_db': [spec1_db, ...]}}
    global_db_min = float('inf')
    global_db_max = float('-inf')

    for frame_number in frames:
        frame_time = frame_number / VIDEO_FRAMERATE
        center_sample = int(frame_time * sr)
        
        start_sample = center_sample - half_window_samples
        end_sample = center_sample + half_window_samples 

        segments = []
        valid_frame = True
        for i, audio_data in enumerate(audios):
            if audio_data is None:
                # print(f"Skipping frame {frame_number} due to missing audio file for mic {i}.") # Reduce verbosity
                valid_frame = False
                break 
                
            slice_start = max(0, start_sample)
            slice_end = min(len(audio_data), end_sample)
            
            pad_start = max(0, -start_sample)
            pad_end = max(0, end_sample - len(audio_data))

            segment = audio_data[slice_start:slice_end]
            
            if pad_start > 0 or pad_end > 0:
                 segment = np.pad(segment, (pad_start, pad_end), 'constant', constant_values=(0, 0))

            if len(segment) != total_window_samples:
                 if len(segment) < total_window_samples:
                     padding_needed = total_window_samples - len(segment)
                     segment = np.pad(segment, (0, padding_needed), 'constant', constant_values=(0, 0))
                 elif len(segment) > total_window_samples:
                     segment = segment[:total_window_samples]

            if len(segment) != total_window_samples:
                 print(f"Warning: Frame {frame_number}, Mic {i}: Segment length issue. Skipping frame.")
                 valid_frame = False
                 break 
                 
            segments.append(segment)

        if not valid_frame:
            continue 

        # Stack segments 
        try:
            multi_channel_audio = np.stack(segments, axis=-1)
        except ValueError as e:
            print(f"Error stacking segments for frame {frame_number}: {e}")
            continue 

        if multi_channel_audio.shape != (total_window_samples, 5):
             print(f"Warning: Frame {frame_number}: Unexpected stacked shape. Skipping frame.")
             continue

        # Define output path and save audio clip
        output_filename = os.path.join(audio_output_directory, f"frame_{frame_number:05d}.wav")
        try:
            sf.write(output_filename, multi_channel_audio, sr)
        except Exception as e:
            print(f"Error writing audio file {output_filename}: {e}")
            continue # Skip spectrogram generation if audio saving failed

        # --- Calculate and Store Spectrograms (dB) ---
        specs_db = []
        frame_valid_for_spec = True
        for i in range(multi_channel_audio.shape[1]): # Iterate through channels
            channel_audio = multi_channel_audio[:, i]
            # Ensure audio is not silent or too short for STFT
            if np.max(np.abs(channel_audio)) < 1e-6 or len(channel_audio) < N_FFT:
                 print(f"Warning: Frame {frame_number}, Mic {i+1}: Audio silent or too short for STFT. Using zeros for spectrogram.")
                 # Create a dummy zero spectrogram of the expected shape
                 # Expected time frames: roughly len(channel_audio) // hop_length
                 n_frames = int(np.ceil(len(channel_audio) / HOP_LENGTH))
                 spec_db = np.full((N_MELS, n_frames), -80.0) # Fill with a low dB value
            else:
                 mel_spec = librosa.feature.melspectrogram(y=channel_audio, sr=sr, n_fft=N_FFT, 
                                                          hop_length=HOP_LENGTH, n_mels=N_MELS,
                                                          fmin=FMIN, fmax=FMAX)
                 spec_db = librosa.power_to_db(mel_spec, ref=np.max) # Use max power as reference
                 
                 # Update global min/max for dB scaling
                 current_min = np.min(spec_db)
                 current_max = np.max(spec_db)
                 if current_min > -np.inf : # Avoid -inf if input was silent
                    global_db_min = min(global_db_min, current_min)
                 global_db_max = max(global_db_max, current_max)

            specs_db.append(spec_db)

        processed_data[frame_number] = {'audio': multi_channel_audio, 'specs_db': specs_db}


    if not processed_data:
        print("No frames were successfully processed to generate spectrograms.")
        sys.exit()

    # Ensure min/max are reasonable, e.g., if only silent frames processed
    if global_db_min == float('inf'): global_db_min = -80.0 # Default min if no valid specs
    if global_db_max == float('-inf'): global_db_max = 0.0 # Default max if no valid specs
    
    # Add a small margin if min and max are identical
    if global_db_min == global_db_max:
        global_db_min -= 1
        global_db_max += 1

    print(f"Global Spectrogram dB range: [{global_db_min:.2f}, {global_db_max:.2f}]")
    print("Generating spectrogram images (Pass 2)...")

    # --- Generate Spectrogram Images (Pass 2) ---
    spectrogram_image_paths = []
    # Use a consistent norm for the colorbar
    norm = mcolors.Normalize(vmin=global_db_min, vmax=global_db_max)
    sm = plt.cm.ScalarMappable(cmap=SPEC_CMAP, norm=norm)
    sm.set_array([]) # Necessary for colorbar without actual data

    sorted_frame_numbers = sorted(processed_data.keys())

    for frame_number in sorted_frame_numbers:
        specs_db = processed_data[frame_number]['specs_db']
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True) # Create 1x5 grid, share Y axis
        fig.suptitle(f'Frame {frame_number:05d} Spectrograms (Mel Scale, dB)')

        for i, spec_db in enumerate(specs_db):
            ax = axes[i]
            img = librosa.display.specshow(spec_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', 
                                           ax=ax, cmap=SPEC_CMAP, vmin=global_db_min, vmax=global_db_max, 
                                           fmin=FMIN, fmax=FMAX)
            ax.set_title(f'Mic {i+1}')
            ax.label_outer() # Hide y-axis labels and ticks for inner plots

        # Add a single colorbar to the figure
        # Position: [left, bottom, width, height] relative to figure
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
        fig.colorbar(sm, cax=cbar_ax, label='dB')

        plt.tight_layout(rect=[0, 0.03, 0.9, 0.95]) # Adjust layout to prevent title overlap and make space for colorbar

        # Save the figure
        image_filename = os.path.join(spectrogram_image_temp_dir, f"spec_frame_{frame_number:05d}.png")
        try:
            plt.savefig(image_filename)
            spectrogram_image_paths.append(image_filename)
        except Exception as e:
            print(f"Error saving spectrogram image {image_filename}: {e}")
        
        plt.close(fig) # Close figure to free memory

    print(f"Generated {len(spectrogram_image_paths)} spectrogram images.")

    # --- Compile Video ---
    if spectrogram_image_paths:
        print(f"Compiling video to {video_output_path} at {VIDEO_FRAMERATE} FPS...")
        try:
            with imageio.get_writer(video_output_path, fps=VIDEO_FRAMERATE) as writer:
                for image_path in spectrogram_image_paths:
                    image = imageio.imread(image_path)
                    writer.append_data(image)
            print("Video compilation successful.")
        except Exception as e:
            print(f"Error compiling video: {e}")
    else:
        print("No spectrogram images were generated, skipping video compilation.")

    # --- Cleanup ---
    print(f"Removing temporary image directory: {spectrogram_image_temp_dir}")
    try:
        shutil.rmtree(spectrogram_image_temp_dir)
    except Exception as e:
        print(f"Error removing temporary directory: {e}")


    print(f"Finished processing.")
    print(f"Audio clips saved to: {audio_output_directory}")
    if spectrogram_image_paths and os.path.exists(video_output_path):
         print(f"Spectrogram video saved to: {video_output_path}")

