# scripts/visualize_results.py
import os
import sys
import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm
import subprocess

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.vsr_model import VSRModel

def main(args):
    print("--- Starting High-Quality Video Visualization ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load the trained Generator model
    try:
        generator = VSRModel(scale_factor=args.scale_factor).to(device)
        generator.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        generator.eval()
        print("Generator model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Open the input video
    if not os.path.exists(args.video_path):
        print(f"Error: Input video not found at {args.video_path}")
        return

    cap = cv2.VideoCapture(args.video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input video: {input_w}x{input_h} @ {fps:.2f} FPS, {total_frames} frames.")

    # 3. Prepare the output video writer using ffmpeg
    output_w = input_w * 3 # For the triple-view comparison
    output_h = input_h
    
    # ffmpeg command for writing video frames from a pipe
    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{output_w}x{output_h}',  # size of one frame
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',  # The input comes from a pipe
        '-an',  # No audio
        '-vcodec', 'libx264',
        '-preset', 'medium',
        '-crf', '20', # Good quality setting
        args.output_path
    ]
    
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # 4. Process video frame by frame
    high_res_previous = None
    prev_gray = None
    
    progress_bar = tqdm(range(total_frames), desc="Processing video")

    for frame_idx in progress_bar:
        ret, frame_hr = cap.read()
        if not ret:
            break

        # Convert frame to tensor and normalize to [0, 1]
        frame_hr_tensor = torch.from_numpy(frame_hr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Create LR version for the model
        lr_h = int(input_h / args.scale_factor)
        lr_w = int(input_w / args.scale_factor)
        frame_lr = cv2.resize(frame_hr, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
        frame_lr_tensor = torch.from_numpy(frame_lr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        # Initialize hidden state and flow for the first frame
        if frame_idx == 0:
            high_res_previous = torch.zeros_like(frame_hr_tensor)
            prev_gray = cv2.cvtColor(frame_lr, cv2.COLOR_BGR2GRAY)
            flow_lr_tensor = torch.zeros(1, 2, lr_h, lr_w).to(device)
        else:
            # Calculate optical flow in real-time (fast Farneback for inference)
            current_gray = cv2.cvtColor(frame_lr, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Convert flow to tensor
            flow_lr_tensor = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).to(device)
            prev_gray = current_gray

        with torch.no_grad():
            # Run the model
            output_hr_tensor = generator(frame_lr_tensor, high_res_previous, flow_lr_tensor)
            
        # Update the history for the next frame
        high_res_previous = output_hr_tensor

        # --- Create comparison frame ---
        # 1. Baseline (Bilinear upscale)
        baseline_np = cv2.resize(frame_lr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        
        # 2. Our Model's Output
        output_np = (output_hr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        
        # 3. Ground Truth (Original)
        ground_truth_np = frame_hr

        # Add text labels to each view
        cv2.putText(baseline_np, 'Bilinear', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(output_np, 'Our Model', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(ground_truth_np, 'Ground Truth', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

        # Concatenate images side-by-side
        final_frame = np.concatenate((baseline_np, output_np, ground_truth_np), axis=1)

        # Write the frame to the ffmpeg pipe
        process.stdin.write(final_frame.tobytes())

    # Clean up
    cap.release()
    process.stdin.close()
    process.wait()
    process.terminate()

    print(f"\n--- Visualization Complete! ---")
    print(f"Output video saved to: {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a high-quality comparison video using a trained VSR model.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file (e.g., a clip from your validation set).")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the trained generator checkpoint (.pth file).")
    parser.add_argument('--output_path', type=str, default='visualization_output.mp4', help="Path to save the output comparison video.")
    parser.add_argument('--scale_factor', type=float, default=7.5, help="The upscale factor the model was trained for (1080p / 144p = 7.5).")

    args = parser.parse_args()
    main(args)