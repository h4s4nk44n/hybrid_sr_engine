import os
import argparse
import numpy as np
import onnx
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.dataset import TemporalFrameDataset # We need the dataset to create a calibrator
from torch.utils.data import DataLoader

class VSRCalibrationDataReader(CalibrationDataReader):
    """
    A custom data reader for providing calibration data to the ONNX quantizer.
    It reads a few batches from our validation dataset.
    """
    def __init__(self, data_loader, scale_factor, num_samples=32):
        self.data_loader_iter = iter(data_loader)
        self.scale_factor = scale_factor
        self.num_samples = num_samples
        self.count = 0

    def get_next(self):
        if self.count >= self.num_samples:
            return None # End of calibration
        
        try:
            batch = next(self.data_loader_iter)
            
            # Use the first frame of the sequence for calibration
            lr_current = batch['imgs_lr'][:, 0, :, :, :].numpy()
            
            # Create dummy previous state and flow
            batch_size, _, lr_h, lr_w = lr_current.shape
            hr_h, hr_w = int(lr_h * self.scale_factor), int(lr_w * self.scale_factor)
            
            hr_previous = np.random.randn(batch_size, 3, hr_h, hr_w).astype(np.float32)
            flow = np.random.randn(batch_size, 2, lr_h, lr_w).astype(np.float32)
            
            self.count += 1
            
            # Return the inputs as a dictionary matching the ONNX model's input names
            return {
                "low_res_current": lr_current,
                "high_res_previous": hr_previous,
                "flow_prev_to_current": flow
            }
        except StopIteration:
            return None

def main(args):
    print("--- Quantizing ONNX Model to INT8 ---")

    # 1. Prepare the Calibration Data Reader
    print("Preparing calibration data reader...")
    if not os.path.exists(args.calibration_data_dir):
        print(f"Error: Calibration data directory not found at '{args.calibration_data_dir}'")
        return
        
    # Use our existing dataset class to load validation data
    calib_dataset = TemporalFrameDataset(root_dir=args.calibration_data_dir, seq_len=1)
    calib_dataloader = DataLoader(calib_dataset, batch_size=1, shuffle=False)
    
    calib_data_reader = VSRCalibrationDataReader(calib_dataloader, scale_factor=7.5, num_samples=args.num_calib_samples)
    
    # 2. Perform Static Quantization
    try:
        print("Starting static quantization. This may take a few minutes...")
        quantize_static(
            model_input=args.input_model,
            model_output=args.output_model,
            calibration_data_reader=calib_data_reader,
            quant_format='QDQ',  # Quantize-Dequantize format, good for CPU/TensorRT
            activation_type=QuantType.QInt8, # Quantize activations to signed 8-bit int
            weight_type=QuantType.QInt8,     # Quantize weights to signed 8-bit int
            per_channel=True, # Use per-channel quantization for weights for better accuracy
            reduce_range=True,
            nodes_to_exclude=[] # Can exclude sensitive nodes if accuracy drops
        )
        print("Static quantization completed successfully.")
        print(f"Quantized model saved to: {args.output_model}")
        
    except Exception as e:
        print(f"An error occurred during quantization: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform static INT8 quantization on an ONNX model.")
    parser.add_argument('--input_model', type=str, required=True, help="Path to the input FP32 ONNX model.")
    parser.add_argument('--output_model', type=str, required=True, help="Path to save the output INT8 ONNX model.")
    parser.add_argument('--calibration_data_dir', type=str, required=True, help="Path to the validation data directory used for calibration.")
    parser.add_argument('--num_calib_samples', type=int, default=100, help="Number of samples to use for calibration.")

    args = parser.parse_args()
    main(args)