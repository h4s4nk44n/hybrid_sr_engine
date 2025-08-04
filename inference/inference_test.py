# inference/benchmark.py
import os
import argparse
import numpy as np
import onnxruntime as ort
import time
import platform
from tqdm import tqdm

def initialize_session(fp32_path, int8_path, num_threads, provider_choice):
    """
    Initializes an ONNX Runtime session with a specific or auto-selected provider.
    """
    print("--- Initializing ONNX Inference Session for Benchmarking ---")
    
    options = ort.SessionOptions()
    options.intra_op_num_threads = num_threads
    
    available_providers = ort.get_available_providers()
    print(f"Available Providers: {available_providers}")

    provider_map = {
        'tensorrt': ('TensorRTExecutionProvider', fp32_path, [{'trt_fp16_enable': True, 'trt_engine_cache_enable': True, 'trt_engine_cache_path': 'engine_cache'}]),
        'directml': ('DmlExecutionProvider', fp32_path, []),
        'rocm': ('ROCmExecutionProvider', fp32_path, []),
        'openvino': ('OpenVINOExecutionProvider', int8_path, []),
        'cuda': ('CUDAExecutionProvider', fp32_path, []),
        'cpu': ('CPUExecutionProvider', int8_path, [])
    }

    # Auto-selection logic
    if provider_choice == 'auto':
        priority_list = []
        if platform.system() == 'Windows':
            priority_list = ['tensorrt', 'directml', 'openvino', 'cuda', 'cpu']
        elif platform.system() == 'Linux':
            priority_list = ['tensorrt', 'rocm', 'openvino', 'cuda', 'cpu']
        else:
            priority_list = ['cpu'] # Fallback for other OS
    else:
        priority_list = [provider_choice]

    # Try to initialize from the priority list
    for provider_name_key in priority_list:
        if provider_map[provider_name_key][0] in available_providers:
            provider, model_path, provider_options = provider_map[provider_name_key]
            print(f"\nAttempting to use {provider} with model '{os.path.basename(model_path)}'...")
            try:
                session = ort.InferenceSession(model_path, providers=[provider], provider_options=provider_options, sess_options=options)
                print(f"Successfully initialized with {provider}.")
                return session, provider
            except Exception as e:
                print(f"  -> Failed to initialize {provider}: {e}")
                if provider_choice != 'auto': # If user explicitly asked for one, fail here
                    return None, None
    
    # If auto-selection fails through all, or a specific choice fails
    print("Could not initialize any preferred provider. No session created.")
    return None, None


def main(args):
    """Main benchmarking function."""
    session, provider_name = initialize_session(args.fp32_model, args.int8_model, args.threads, args.provider)
    if session is None:
        print("Exiting benchmark due to session initialization failure.")
        return

    # --- Prepare Dummy Data ---
    print("\n--- Preparing Dummy Data ---")
    scale_factor = args.scale_factor
    lr_h = int(1080 / scale_factor)
    lr_w = int(1920 / scale_factor)
    hr_h, hr_w = 1080, 1920

    # Create random NumPy arrays that mimic real input tensors
    lr_current = np.random.randn(1, 3, lr_h, lr_w).astype(np.float32)
    hr_previous = np.random.randn(1, 3, hr_h, hr_w).astype(np.float32)
    flow = np.random.randn(1, 2, lr_h, lr_w).astype(np.float32)

    input_names = [inp.name for inp in session.get_inputs()]
    inputs = {
        input_names[0]: lr_current,
        input_names[1]: hr_previous,
        input_names[2]: flow
    }
    print(f"Input tensors created with shapes: LR({lr_current.shape}), HR_prev({hr_previous.shape}), Flow({flow.shape})")

    # --- Warm-up Run ---
    print("\n--- Performing Warm-up Runs ---")
    for _ in range(args.warmup_runs):
        session.run(None, inputs)
    print("Warm-up complete.")

    # --- Benchmarking Loop ---
    print(f"\n--- Running Benchmark for {args.iterations} Iterations ---")
    timings = []
    
    for _ in tqdm(range(args.iterations), desc="Benchmarking"):
        start_time = time.perf_counter()
        
        # Run inference. The output is a new HR frame.
        output_hr_tensor = session.run(None, inputs)[0]
        
        end_time = time.perf_counter()
        timings.append(end_time - start_time)
        
        # In a real recurrent scenario, the output would become the next input.
        # For a pure benchmark, we can just feed the same data, but this is more realistic.
        inputs[input_names[1]] = output_hr_tensor

    # --- Calculate and Print Results ---
    total_time = sum(timings)
    avg_time_ms = (total_time / args.iterations) * 1000
    fps = args.iterations / total_time

    print("\n--- Benchmark Results ---")
    print(f"Execution Provider: {provider_name}")
    print(f"Model (INT8):       {os.path.basename(args.int8_model)}" if 'CPU' in provider_name or 'OpenVINO' in provider_name else f"Model (FP32):       {os.path.basename(args.fp32_model)}")
    print(f"Total Iterations:   {args.iterations}")
    print(f"Average Time:       {avg_time_ms:.3f} ms")
    print(f"Frames Per Second:  {fps:.2f} FPS")
    print("-------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark the inference speed of VSR ONNX models.")
    parser.add_argument('--fp32_model', type=str, required=True, help="Path to the FP32 ONNX model.")
    parser.add_argument('--int8_model', type=str, required=True, help="Path to the INT8 Quantized ONNX model.")
    parser.add_argument('--provider', type=str, default='auto', choices=['auto', 'tensorrt', 'directml', 'rocm', 'openvino', 'cuda', 'cpu'], help="Specify an execution provider, or 'auto' for priority-based selection.")
    parser.add_argument('--iterations', type=int, default=500, help="Number of iterations to run the benchmark.")
    parser.add_argument('--warmup_runs', type=int, default=20, help="Number of warm-up runs before starting the benchmark.")
    parser.add_argument('--threads', type=int, default=4, help="Number of threads for ONNX Runtime to use (primarily for CPU).")
    parser.add_argument('--scale_factor', type=float, default=7.5, help="Upscale factor of the model.")

    args = parser.parse_args()
    main(args)