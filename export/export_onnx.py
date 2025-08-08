# export/export_onnx.py
import os
import argparse
import torch
import numpy as np
import onnx
import onnxruntime as ort # Import onnxruntime for verification
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.vsr_model import VSRModel

def main(args):
    print("--- Exporting PyTorch Model to ONNX (FP32) ---")

    device = torch.device('cpu') # ONNX export should be done on CPU

    # 1. Load the trained Generator model
    try:
        model = VSRModel(scale_factor=args.scale_factor).to(device)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        model.eval()
        print(f"Generator model loaded successfully from: {args.checkpoint_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Create dummy inputs
    lr_h = int(1080 / args.scale_factor)
    lr_w = int(1920 / args.scale_factor)
    
    dummy_lr_current = torch.randn(1, 3, lr_h, lr_w, device=device)
    dummy_hr_previous = torch.randn(1, 3, 1080, 1920, device=device)
    dummy_flow = torch.randn(1, 2, lr_h, lr_w, device=device)
    
    dummy_input = (dummy_lr_current, dummy_hr_previous, dummy_flow)

    # 3. Define input and output names
    input_names = ["low_res_current", "high_res_previous", "flow_prev_to_current"]
    output_names = ["output_high_res"]
    
    # 4. Export the model
    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"Exporting to ONNX format at: {args.output_path}")
        torch.onnx.export(
            model, dummy_input, args.output_path,
            verbose=False,
            input_names=input_names, output_names=output_names,
            opset_version=14,
            dynamic_axes={
                "low_res_current": {0: "batch_size"},
                "high_res_previous": {0: "batch_size"},
                "flow_prev_to_current": {0: "batch_size"},
                "output_high_res": {0: "batch_size"}
            }
        )
        print("ONNX export completed successfully.")

        # --- UPGRADE: Numerical Verification Step ---
        print("\n--- Verifying ONNX Model ---")
        
        # 1. Structural Check
        onnx_model = onnx.load(args.output_path)
        onnx.checker.check_model(onnx_model)
        print("Step 1/2: Structural check passed.")

        # 2. Numerical Check
        # Run inference with both models and compare the output
        with torch.no_grad():
            pytorch_output = model(*dummy_input).numpy()

        ort_session = ort.InferenceSession(args.output_path)
        ort_inputs = {
            input_names[0]: dummy_lr_current.numpy(),
            input_names[1]: dummy_hr_previous.numpy(),
            input_names[2]: dummy_flow.numpy()
        }
        onnx_output = ort_session.run(None, ort_inputs)[0]

        # Compare outputs. They should be very close.
        np.testing.assert_allclose(pytorch_output, onnx_output, rtol=1e-3, atol=1e-5)
        print("Step 2/2: Numerical check passed. PyTorch and ONNX outputs are close.")
        print("\nONNX model verification successful.")

    except Exception as e:
        print(f"An error occurred during export or verification: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export a trained VSR Generator to ONNX format.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the trained generator checkpoint (.pth file).")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output ONNX model (e.g., 'onnx_models/vsr_model_fp32.onnx').")
    parser.add_argument('--scale_factor', type=float, default=7.5, help="The upscale factor the model was trained for.")

    args = parser.parse_args()
    main(args)