# inference/inference_demo.py
import os
import cv2
import numpy as np
import onnxruntime as ort
import time
import threading
import queue
import platform
import argparse

# --- Utility Functions & Classes ---
class FrameRateMonitor:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.last_time = time.perf_counter()
        self.fps = 0

    def tick(self):
        current_time = time.perf_counter()
        delta = current_time - self.last_time
        self.last_time = current_time
        if delta > 0:
            current_fps = 1.0 / delta
            self.fps = self.alpha * current_fps + (1 - self.alpha) * self.fps
        return self.fps

def initialize_session(fp32_path, int8_path, num_threads):
    """
    Intelligently selects the best execution provider and model file.
    Follows our agreed-upon universal strategy.
    """
    print("--- Initializing ONNX Inference Session ---")
    
    options = ort.SessionOptions()
    options.intra_op_num_threads = num_threads
    
    available_providers = ort.get_available_providers()
    print(f"Available Providers: {available_providers}")

    # Priority List: TensorRT > DirectML > ROCm > OpenVINO > CPU
    # 1. TensorRT (NVIDIA)
    if 'TensorRTExecutionProvider' in available_providers:
        print("Attempting to use TensorRTExecutionProvider...")
        try:
            # TensorRT performs its own optimizations, including INT8. Feed it the FP32 model.
            session = ort.InferenceSession(fp32_path, providers=['TensorRTExecutionProvider'], provider_options=[{'trt_fp16_enable': True, 'trt_int8_enable': True, 'trt_engine_cache_enable': True, 'trt_engine_cache_path': 'engine_cache'}], sess_options=options)
            print("Successfully initialized with TensorRT.")
            return session, 'TensorRT'
        except Exception as e:
            print(f"TensorRT failed: {e}. Trying next provider.")

    # 2. DirectML (AMD/Intel/NVIDIA on Windows)
    if platform.system() == 'Windows' and 'DmlExecutionProvider' in available_providers:
        print("Attempting to use DmlExecutionProvider...")
        try:
            # DirectML works best with the FP32 model.
            session = ort.InferenceSession(fp32_path, providers=['DmlExecutionProvider'], sess_options=options)
            print("Successfully initialized with DirectML.")
            return session, 'DirectML'
        except Exception as e:
            print(f"DirectML failed: {e}. Trying next provider.")
            
    # 3. ROCm (AMD on Linux)
    if platform.system() == 'Linux' and 'ROCmExecutionProvider' in available_providers:
        print("Attempting to use ROCmExecutionProvider...")
        try:
            session = ort.InferenceSession(fp32_path, providers=['ROCmExecutionProvider'], sess_options=options)
            print("Successfully initialized with ROCm.")
            return session, 'ROCm'
        except Exception as e:
            print(f"ROCm failed: {e}. Trying next provider.")

    # 4. OpenVINO (Intel)
    if 'OpenVINOExecutionProvider' in available_providers:
        print("Attempting to use OpenVINOExecutionProvider...")
        try:
            # OpenVINO is optimized for our pre-quantized INT8 model.
            session = ort.InferenceSession(int8_path, providers=['OpenVINOExecutionProvider'], sess_options=options)
            print("Successfully initialized with OpenVINO.")
            return session, 'OpenVINO'
        except Exception as e:
            print(f"OpenVINO failed: {e}. Trying next provider.")

    # 5. Fallback CPU
    print("Falling back to CPUExecutionProvider...")
    # The CPU provider is fastest with the INT8 quantized model.
    session = ort.InferenceSession(int8_path, providers=['CPUExecutionProvider'], sess_options=options)
    print("Successfully initialized with CPU.")
    return session, 'CPU'

# --- Threading Loops ---
def capture_loop(cap, infer_q, display_q, stop_event):
    """Reads frames from the source and puts them into queues."""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("End of video source.")
            stop_event.set()
            break
        
        # Non-blocking put, discard old frame if queue is full
        try:
            infer_q.put_nowait(frame)
            display_q.put_nowait(frame)
        except queue.Full:
            pass

def inference_loop(session, scale_factor, infer_q, sr_frame_buffer, sr_fps_monitor, stop_event):
    """Performs stateful, recurrent inference on incoming frames."""
    # Get model input names from the session
    input_names = [inp.name for inp in session.get_inputs()]

    # Initialize state variables
    high_res_previous = None
    prev_gray = None

    while not stop_event.is_set():
        try:
            frame_bgr = infer_q.get(timeout=1)
        except queue.Empty:
            continue
            
        # Prepare Low-Resolution Frame
        hr_h, hr_w, _ = frame_bgr.shape
        lr_h, lr_w = int(hr_h / scale_factor), int(hr_w / scale_factor)
        frame_lr_bgr = cv2.resize(frame_bgr, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
        
        # Prepare Tensor for Model
        frame_lr_rgb = cv2.cvtColor(frame_lr_bgr, cv2.COLOR_BGR2RGB)
        frame_lr_tensor = (frame_lr_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        # Initialize or calculate optical flow
        if prev_gray is None:
            # First frame
            prev_gray = cv2.cvtColor(frame_lr_bgr, cv2.COLOR_BGR2GRAY)
            flow_tensor = np.zeros((1, 2, lr_h, lr_w), dtype=np.float32)
            high_res_previous = np.zeros((1, 3, hr_h, hr_w), dtype=np.float32)
        else:
            current_gray = cv2.cvtColor(frame_lr_bgr, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_tensor = flow.transpose(2, 0, 1)[None, ...]
            prev_gray = current_gray

        # Run Inference
        inputs = {
            input_names[0]: frame_lr_tensor,
            input_names[1]: high_res_previous,
            input_names[2]: flow_tensor
        }
        output_hr_tensor = session.run(None, inputs)[0]
        
        # Update state for the next frame
        high_res_previous = output_hr_tensor

        # Update shared buffer for display
        sr_frame = (output_hr_tensor[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        sr_frame_buffer['frame'] = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)
        sr_fps_monitor.tick()

# --- Main Application ---
def main(args):
    # --- Initialization ---
    session, provider_name = initialize_session(args.fp32_model, args.int8_model, args.threads)
    
    # --- Data Source ---
    try:
        source_id = int(args.input)
        cap = cv2.VideoCapture(source_id)
    except ValueError:
        if not os.path.exists(args.input):
            print(f"Error: Video file not found at {args.input}")
            return
        cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return
        
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # --- Shared Resources & Threads ---
    infer_q = queue.Queue(maxsize=1)
    display_q = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    
    sr_frame_buffer = {'frame': np.zeros((source_h, source_w, 3), dtype=np.uint8)}
    sr_fps_monitor = FrameRateMonitor()

    capture_thread = threading.Thread(target=capture_loop, args=(cap, infer_q, display_q, stop_event), daemon=True)
    inference_thread = threading.Thread(target=inference_loop, args=(session, args.scale_factor, infer_q, sr_frame_buffer, sr_fps_monitor, stop_event), daemon=True)

    capture_thread.start()
    inference_thread.start()
    
    # --- Display Loop ---
    cv2.namedWindow("AI Video Upscaler", cv2.WINDOW_NORMAL)

    while not stop_event.is_set():
        try:
            original_frame = display_q.get(timeout=1)
        except queue.Empty:
            if not capture_thread.is_alive():
                break
            continue

        lr_display = cv2.resize(original_frame, (source_w // 2, source_h), interpolation=cv2.INTER_NEAREST)
        sr_display = cv2.resize(sr_frame_buffer['frame'], (source_w // 2, source_h), interpolation=cv2.INTER_LINEAR)

        # Create canvas
        canvas = np.concatenate((lr_display, sr_display), axis=1)
        
        # Add text overlays
        cv2.putText(canvas, "Original (for reference)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(canvas, f"AI Upscaled ({provider_name})", (source_w // 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(canvas, f"{sr_fps_monitor.fps:.1f} FPS", (source_w // 2 + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("AI Video Upscaler", canvas)

        if cv2.waitKey(1) in (27, ord('q')):
            stop_event.set()
            break

    # --- Cleanup ---
    capture_thread.join(timeout=2)
    inference_thread.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-time AI Video Upscaler using ONNX Runtime.")
    parser.add_argument('--fp32_model', type=str, required=True, help="Path to the FP32 ONNX model (for TensorRT, DirectML, ROCm).")
    parser.add_argument('--int8_model', type=str, required=True, help="Path to the INT8 Quantized ONNX model (for OpenVINO, CPU).")
    parser.add_argument('--input', type=str, default='0', help="Input source. Can be a webcam ID (e.g., '0') or a path to a video file.")
    parser.add_argument('--scale_factor', type=float, default=7.5, help="Upscale factor of the model.")
    parser.add_argument('--threads', type=int, default=4, help="Number of threads for ONNX Runtime to use.")
    
    args = parser.parse_args()
    main(args)