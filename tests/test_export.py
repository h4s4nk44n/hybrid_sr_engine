import unittest
import os
import torch
import numpy as np
import onnxruntime as ort
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.vsr_model import VSRModel
from export.export_onnx import main as export_main # Assuming the export script has a main function

class TestONNXExport(unittest.TestCase):

    def setUp(self):
        """Set up a dummy model and paths for testing."""
        self.scale_factor = 7.5
        self.dummy_checkpoint_path = "temp_dummy_generator.pth"
        self.onnx_output_path = "temp_dummy_model.onnx"
        
        # Create and save a dummy PyTorch model checkpoint
        self.model = VSRModel(scale_factor=self.scale_factor)
        torch.save(self.model.state_dict(), self.dummy_checkpoint_path)

    def tearDown(self):
        """Remove temporary files."""
        if os.path.exists(self.dummy_checkpoint_path):
            os.remove(self.dummy_checkpoint_path)
        if os.path.exists(self.onnx_output_path):
            os.remove(self.onnx_output_path)

    def test_export_and_inference(self):
        """Tests if the ONNX export runs and if the model produces similar outputs."""
        # 1. Run the export script
        # We create a dummy args object to pass to the main function of the export script
        class Args:
            checkpoint_path = self.dummy_checkpoint_path
            output_path = self.onnx_output_path
            scale_factor = self.scale_factor
        
        export_main(Args())

        # Check if the ONNX file was created
        self.assertTrue(os.path.exists(self.onnx_output_path))

        # 2. Run inference with both models and compare outputs
        self.model.eval()
        
        # Create dummy inputs
        lr_h, lr_w = 144, 256
        hr_h, hr_w = 1080, 1920
        dummy_lr = torch.randn(1, 3, lr_h, lr_w)
        dummy_hr_prev = torch.randn(1, 3, hr_h, hr_w)
        dummy_flow = torch.randn(1, 2, lr_h, lr_w)

        # PyTorch inference
        with torch.no_grad():
            pytorch_output = self.model(dummy_lr, dummy_hr_prev, dummy_flow).numpy()

        # ONNX inference
        ort_session = ort.InferenceSession(self.onnx_output_path)
        input_names = [inp.name for inp in ort_session.get_inputs()]
        inputs = {
            input_names[0]: dummy_lr.numpy(),
            input_names[1]: dummy_hr_prev.numpy(),
            input_names[2]: dummy_flow.numpy()
        }
        onnx_output = ort_session.run(None, inputs)[0]

        # Compare the outputs
        # They won't be exactly identical due to floating point differences,
        # so we check if they are "close enough".
        np.testing.assert_allclose(pytorch_output, onnx_output, rtol=1e-3, atol=1e-5)
        print("\nPyTorch and ONNX model outputs are numerically close. Test passed.")


if __name__ == '__main__':
    unittest.main()