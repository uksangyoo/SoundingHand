#!/usr/bin/env python3
import os
import re
import glob
import json
import argparse

import torch
import torch.nn.functional as F
import torchaudio
from scipy.io import wavfile

def get_audio_length(wav_path):
    sr, data = wavfile.read(wav_path)
    return len(data) / sr

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
    print("start_cam:", start_cam)
    print("n_cam:", n_cam)
    return start_cam, n_cam

def get_mic0_start(json_path):
    with open(json_path) as f:
        d = json.load(f)
    for rec in d.get('recordings', []):
        if rec.get('mic_number') == 0:
            return rec['start_time_ms'] / 1000.0
    raise RuntimeError("mic0 entry not found")

def crop_wav(input_path, start_time, seg_len):
    waveform, sample_rate = torchaudio.load(input_path)  # [channels, total_samples]
    start_sample = int(round(start_time * sample_rate))
    num_samples  = int(round(seg_len      * sample_rate))
    end_sample   = start_sample + num_samples

    segment = waveform[:, start_sample:end_sample]
    return segment, sample_rate

def slice_audio_for_trial(
    mic_raw_root, sync_log_root, object_name, trial_name, seg_len=1.0/30.0, output_root="/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA"
):
    """
    Slices audio for a single trial and saves to the correct output directory.
    """
    mic_paths = get_audio_paths(mic_raw_root, trial_name)
    if not mic_paths:
        print(f"No mic files found for {object_name} {trial_name}! Skipping.")
        return

    # get sync info
    vid_start, num_frames = get_video_info(os.path.join(sync_log_root, f"{trial_name}_cam_sync.json"))
    audio0_start = get_mic0_start(os.path.join(sync_log_root, f"{trial_name}_mic_sync.json"))
    audio_offset = 6.5e-3
    frame_len = 1.0 / 30.0

    # compute how many full segments each mic can provide
    mic_indices = sorted(mic_paths.keys())
    local_starts = {}
    counts = {}
    for mic in mic_indices:
        fn = mic_paths[mic]
        total_dur = get_audio_length(fn)
        ls = vid_start - audio0_start - seg_len/2 + mic * audio_offset
        local_starts[mic] = ls

        cnt = 0
        for j in range(num_frames - 1):
            if ls + j * frame_len + seg_len > total_dur:
                break
            cnt += 1
        counts[mic] = cnt
        print(f"mic{mic}: can produce {cnt} segments")

    min_frames = min(counts.values())
    if min_frames == 0:
        print(f"No overlapping segments for {object_name} {trial_name}, skipping.")
        return

    # prepare output directory
    comb_dir = os.path.join(output_root, object_name, trial_name, "audio")
    os.makedirs(comb_dir, exist_ok=True)
    print(f"Slicing audio for {object_name} {trial_name} into {comb_dir}")
    # slice and save
    for j in range(min_frames):
        chans = []
        sr0 = None
        for mic in mic_indices:
            fn = mic_paths[mic]
            ls = local_starts[mic]
            start_time = ls + j * frame_len
            wav, sr = crop_wav(fn, start_time, seg_len)
            if sr0 is None:
                sr0 = sr
            elif sr != sr0:
                raise RuntimeError("Sample rates differ across mics!")
            chans.append(wav)  # [1, T]

        multi = torch.cat(chans, dim=0)  # [num_mics, T]
        out_fp = os.path.join(comb_dir, f"{trial_name}_frame{j:04d}.wav")
        torchaudio.save(out_fp, multi, sr0)

    print(f"Wrote {min_frames} {len(mic_indices)}-channel WAVs to {comb_dir}")


def batch_slice_audio():
    DATA_ROOT = "/media/frida/Extreme SSD/sounding_hand/yuemin"
    OUTPUT_ROOT = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA"
    object_dirs = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    for object_name in object_dirs:
        object_path = os.path.join(DATA_ROOT, object_name)
        mic_raw_root = os.path.join(object_path, "mic_raw")
        sync_log_root = os.path.join(object_path, "sync_log")
        if not os.path.isdir(mic_raw_root) or not os.path.isdir(sync_log_root):
            continue
        trial_dirs = [d for d in os.listdir(mic_raw_root) if os.path.isdir(os.path.join(mic_raw_root, d))]
        for trial_name in sorted(trial_dirs):
            print(f"\nProcessing {object_name} {trial_name}")
            try:
                slice_audio_for_trial(mic_raw_root, sync_log_root, object_name, trial_name)
            except Exception as e:
                print(f"Error processing {object_name} {trial_name}: {e}")

if __name__ == "__main__":
    batch_slice_audio()
