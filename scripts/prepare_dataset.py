# scripts/prepare_dataset.py
import os
import sys
import argparse
import glob
import random
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Add project root to sys.path to allow imports from other folders
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Integrated Optical Flow Logic
import cv2
import numpy as np

def process_frame_pair(args_tuple):
    """Worker function to compute optical flow for a single pair of frames."""
    prev_path, next_path, output_dir, algorithm = args_tuple
    
    try:
        # Assumes filenames are like 'frame_00000001.png'
        prev_idx_str = os.path.splitext(os.path.basename(prev_path))[0].split('_')[-1]
        next_idx_str = os.path.splitext(os.path.basename(next_path))[0].split('_')[-1]
        output_filename = f"flow_{int(prev_idx_str):08d}_{int(next_idx_str):08d}.npy"
    except (ValueError, IndexError):
        return ("failed", f"invalid filename format for {os.path.basename(prev_path)}")
    
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        return ("skipped", output_path)

    try:
        prev_frame = cv2.imread(prev_path)
        next_frame = cv2.imread(next_path)
        if prev_frame is None or next_frame is None:
            return ("failed", f"could not read {os.path.basename(prev_path)} or {os.path.basename(next_path)}")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        if algorithm == 'tvl1':
            tvl1 = cv2.optflow.createOptFlow_DualTVL1()
            flow = tvl1.calc(prev_gray, next_gray, None)
        else: # farneback
            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
        np.save(output_path, flow)
        return ("success", output_path)
    except Exception as e:
        return ("failed", str(e))


def main(args):
    """Master orchestrator for the entire data preparation pipeline."""
    print("--- Starting Full Dataset Preparation ---")
    
    # 1. Find all video files
    print(f"Searching for videos in: {args.video_dir}")
    video_extensions = ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv')
    source_videos = []
    for ext in video_extensions:
        source_videos.extend(glob.glob(os.path.join(args.video_dir, f'**/*{ext}'), recursive=True))

    if not source_videos:
        print(f"Error: No video files found in '{args.video_dir}'.")
        return

    print(f"Found {len(source_videos)} total video files.")

    # 2. Split into train and validation sets
    random.shuffle(source_videos)
    split_index = int(len(source_videos) * args.val_split)
    val_videos = source_videos[:split_index]
    train_videos = source_videos[split_index:]

    print(f"Splitting dataset: {len(train_videos)} training videos, {len(val_videos)} validation videos.")
    
    sets_to_process = [('train', train_videos), ('validation', val_videos)]
    
    # Ensure root output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    for set_name, video_list in sets_to_process:
        if not video_list:
            print(f"Skipping empty '{set_name}' set.")
            continue
            
        print(f"\n--- Processing '{set_name}' Set ({len(video_list)} videos) ---")
        set_output_dir = os.path.join(args.output_dir, set_name)
        os.makedirs(set_output_dir, exist_ok=True)

        for i, video_path in enumerate(video_list):
            # Create a clean, unique folder name for each video
            video_basename = f"video_{i:04d}_{os.path.splitext(os.path.basename(video_path))[0]}"
            video_frame_dir = os.path.join(set_output_dir, video_basename)
            
            print(f"\n[{i+1}/{len(video_list)}] Processing Video: {os.path.basename(video_path)}")
            print(f"  -> Output directory: {video_frame_dir}")

            if os.path.exists(video_frame_dir) and not args.overwrite:
                print("  -> Directory already exists and overwrite is False. Skipping.")
                continue
            
            os.makedirs(video_frame_dir, exist_ok=True)

            # 3. Extract frames using ffmpeg
            print("  Step 1/2: Extracting frames...")
            # Using %08d for padding to support a very large number of frames
            output_pattern = os.path.join(video_frame_dir, "frame_%08d.png")
            try:
                command = [
                    'ffmpeg', '-i', video_path, '-q:v', '2', '-vf', f'fps={args.fps}',
                    output_pattern
                ]
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            except Exception as e:
                print(f"  -> ERROR extracting frames from {video_path}. Make sure ffmpeg is installed.")
                print(f"  -> Error message: {getattr(e, 'stderr', e)}")
                continue

            # 4. Calculate optical flow for the extracted frames
            print("  Step 2/2: Calculating optical flow...")
            frame_files = sorted(glob.glob(os.path.join(video_frame_dir, '*.png')))
            if len(frame_files) < 2:
                print("  -> Not enough frames to calculate flow. Skipping.")
                continue

            flow_tasks = [(frame_files[j], frame_files[j+1], video_frame_dir, args.flow_algorithm) for j in range(len(frame_files) - 1)]
            
            with Pool(processes=args.workers) as pool:
                list(tqdm(pool.imap_unordered(process_frame_pair, flow_tasks), total=len(flow_tasks), desc="    Calculating Flow"))

    print("\n--- Dataset Preparation Complete! ---")
    print(f"Your dataset is ready for training at: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automated Video to Dataset Preparation Script")
    parser.add_argument('--video_dir', type=str, required=True, help="Directory containing the source 1080p videos.")
    parser.add_argument('--output_dir', type=str, default='data/frames', help="Root directory to save the final dataset (e.g., 'data/frames').")
    parser.add_argument('--val_split', type=float, default=0.1, help="Fraction of videos for validation (default: 0.1 for 10%).")
    parser.add_argument('--fps', type=int, default=30, help="Frames per second to extract.")
    parser.add_gument('--flow_algorithm', type=str, default='tvl1', choices=['tvl1', 'farneback'], help="Optical flow algorithm. 'tvl1' is higher quality but slower.")
    parser.add_argument('--workers', type=int, default=cpu_count(), help=f"Number of CPU cores for parallel processing (default: {cpu_count()}).")
    parser.add_argument('--overwrite', action='store_true', help="If specified, overwrite existing video frame directories.")
    
    args = parser.parse_args()
    main(args)