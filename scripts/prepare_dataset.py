# scripts/prepare_dataset.py
import os
import sys
import argparse
import glob
import random
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil

# Add project root to sys.path
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
        if prev_frame is None or next_frame is None: return ("failed", f"could not read frames")
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        if algorithm == 'tvl1':
            tvl1 = cv2.optflow.createOptFlow_DualTVL1()
            flow = tvl1.calc(prev_gray, next_gray, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        np.save(output_path, flow)
        return ("success", output_path)
    except Exception as e:
        return ("failed", str(e))

def process_video_frames(video_frame_dir, algorithm, workers):
    """Helper function to calculate optical flow for all frames in a directory."""
    print(f"  Calculating optical flow for directory: {os.path.basename(video_frame_dir)}...")
    frame_files = sorted(glob.glob(os.path.join(video_frame_dir, '*.png')))
    if len(frame_files) < 2:
        print("  -> Not enough frames to calculate flow. Skipping.")
        return
    flow_tasks = [(frame_files[j], frame_files[j+1], video_frame_dir, algorithm) for j in range(len(frame_files) - 1)]
    with Pool(processes=workers) as pool:
        list(tqdm(pool.imap_unordered(process_frame_pair, flow_tasks), total=len(flow_tasks), desc="    Calculating Flow"))

def main(args):
    """Master orchestrator for the entire data preparation pipeline."""
    print("--- Starting Full Dataset Preparation ---")
    source_videos = []
    for ext in ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv'):
        source_videos.extend(glob.glob(os.path.join(args.video_dir, f'**/*{ext}'), recursive=True))

    if not source_videos:
        print(f"Error: No video files found in '{args.video_dir}'.")
        return

    print(f"Found {len(source_videos)} total video files.")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- NEW LOGIC: Handle single video vs. multiple videos differently ---
    if len(source_videos) == 1:
        # --- SINGLE VIDEO LOGIC ---
        video_path = source_videos[0]
        video_name = os.path.splitext(os.path.basename(video_path))[0].replace(' ', '_')
        print(f"\nProcessing single video: {video_name}")
        print("This video will be split into training and validation frame sets.")

        # 1. Extract all frames into a temporary master folder
        master_frame_dir = os.path.join(args.output_dir, f"{video_name}_master_frames")
        if os.path.exists(master_frame_dir) and not args.overwrite:
            print(f"Master frame directory '{master_frame_dir}' already exists. Using existing frames.")
        else:
            os.makedirs(master_frame_dir, exist_ok=True)
            print("  Step 1/3: Extracting all frames...")
            output_pattern = os.path.join(master_frame_dir, "frame_%08d.png")
            try:
                command = ['ffmpeg', '-i', video_path, '-q:v', '2', '-vf', f'fps={args.fps}', output_pattern]
                subprocess.run(command, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"  -> ERROR extracting frames: {e.stderr}")
                return
        
        # 2. Split the list of frames
        all_frames = sorted(glob.glob(os.path.join(master_frame_dir, '*.png')))
        if not all_frames:
            print("Error: No frames were extracted from the video.")
            return

        split_index = int(len(all_frames) * (1 - args.val_split))
        train_frames = all_frames[:split_index]
        val_frames = all_frames[split_index:]

        if not val_frames:
            print("WARNING: Validation split is 0%. All frames will be used for training.")
        
        print(f"Splitting frames: {len(train_frames)} for training, {len(val_frames)} for validation.")
        
        # 3. Create final directories and process each set
        train_dir = os.path.join(args.output_dir, 'train', video_name)
        val_dir = os.path.join(args.output_dir, 'validation', video_name)
        
        print("  Step 2/3: Copying frames to final directories...")
        os.makedirs(train_dir, exist_ok=True)
        for frame_path in tqdm(train_frames, desc="  Copying train frames"):
            shutil.copy(frame_path, train_dir)
        
        if val_frames:
            os.makedirs(val_dir, exist_ok=True)
            for frame_path in tqdm(val_frames, desc="  Copying val frames"):
                shutil.copy(frame_path, val_dir)

        print("  Step 3/3: Calculating optical flow for each set...")
        process_video_frames(train_dir, args.flow_algorithm, args.workers)
        if val_frames:
            process_video_frames(val_dir, args.flow_algorithm, args.workers)

    else:
        # --- MULTIPLE VIDEO LOGIC (Original logic, slightly improved) ---
        random.shuffle(source_videos)
        split_index = int(len(source_videos) * args.val_split)
        if args.val_split > 0 and split_index == 0:
            split_index = 1
        val_videos = source_videos[:split_index]
        train_videos = source_videos[split_index:]

        print(f"Splitting videos: {len(train_videos)} for training, {len(val_videos)} for validation.")
        sets_to_process = [('train', train_videos), ('validation', val_videos)]

        for set_name, video_list in sets_to_process:
            if not video_list: continue
            print(f"\n--- Processing '{set_name}' Set ({len(video_list)} videos) ---")
            set_output_dir = os.path.join(args.output_dir, set_name)
            os.makedirs(set_output_dir, exist_ok=True)
            for i, video_path in enumerate(video_list):
                video_basename = f"video_{i:04d}_{os.path.splitext(os.path.basename(video_path))[0].replace(' ', '_')}"
                video_frame_dir = os.path.join(set_output_dir, video_basename)
                print(f"\n[{i+1}/{len(video_list)}] Processing Video: {os.path.basename(video_path)}")
                if os.path.exists(video_frame_dir) and not args.overwrite:
                    print("  -> Directory already exists. Skipping.")
                    continue
                os.makedirs(video_frame_dir, exist_ok=True)
                print("  Step 1/2: Extracting frames...")
                output_pattern = os.path.join(video_frame_dir, "frame_%08d.png")
                try:
                    command = ['ffmpeg', '-i', video_path, '-q:v', '2', '-vf', f'fps={args.fps}', output_pattern]
                    subprocess.run(command, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"  -> ERROR extracting frames: {e.stderr}")
                    continue
                process_video_frames(video_frame_dir, args.flow_algorithm, args.workers)

    print("\n--- Dataset Preparation Complete! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automated Video to Dataset Preparation Script")
    parser.add_argument('--video_dir', type=str, required=True, help="Directory containing the source 1080p videos.")
    parser.add_argument('--output_dir', type=str, default='data/frames', help="Root directory to save the final dataset.")
    parser.add_argument('--val_split', type=float, default=0.1, help="Fraction of videos/frames for validation (0.0 to 1.0).")
    parser.add_argument('--fps', type=int, default=30, help="Frames per second to extract from the videos.")
    parser.add_argument('--flow_algorithm', type=str, default='tvl1', choices=['tvl1', 'farneback'], help="Optical flow algorithm.")
    parser.add_argument('--workers', type=int, default=cpu_count(), help=f"Number of CPU cores for parallel processing.")
    parser.add_argument('--overwrite', action='store_true', help="If specified, overwrite existing directories.")
    
    args = parser.parse_args()
    main(args)