import unittest
import os
import shutil
import numpy as np
import cv2
import torch
import sys

# Add project root to sys.path to import from the 'data' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.dataset import TemporalFrameDataset

class TestTemporalFrameDataset(unittest.TestCase):
    def setUp(self):
        """Creates a temporary directory with a dummy dataset for testing."""
        self.test_root_dir = "temp_test_dataset"
        self.video_dir = os.path.join(self.test_root_dir, "dummy_video_01")
        os.makedirs(self.video_dir, exist_ok=True)
        
        self.hr_h, self.hr_w = 1080, 1920
        self.num_frames = 10
        self.seq_len = 8

        # Create dummy HR frames
        for i in range(self.num_frames):
            dummy_frame = np.random.randint(0, 256, (self.hr_h, self.hr_w, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(self.video_dir, f"frame_{i:08d}.png"), dummy_frame)

        # Create dummy HR optical flow files
        for i in range(1, self.num_frames):
            dummy_flow = np.random.randn(self.hr_h, self.hr_w, 2).astype(np.float32)
            np.save(os.path.join(self.video_dir, f"flow_{i-1:08d}_{i:08d}.npy"), dummy_flow)

    def tearDown(self):
        """Removes the temporary directory after the test is complete."""
        if os.path.exists(self.test_root_dir):
            shutil.rmtree(self.test_root_dir)

    def test_dataset_output(self):
        """Tests the shape, type, and values of the dataset output."""
        dataset = TemporalFrameDataset(root_dir=self.test_root_dir, seq_len=self.seq_len)
        
        # We should have (num_frames - seq_len + 1) samples
        self.assertEqual(len(dataset), self.num_frames - self.seq_len + 1)

        # Get the first sample
        sample = dataset[0]

        # Check if all required keys are present
        self.assertIn("imgs_lr", sample)
        self.assertIn("imgs_hr", sample)
        self.assertIn("flows", sample)

        # Check tensor shapes
        imgs_lr = sample['imgs_lr']
        imgs_hr = sample['imgs_hr']
        flows = sample['flows']
        
        lr_h, lr_w = 144, 256
        self.assertEqual(imgs_lr.shape, (self.seq_len, 3, lr_h, lr_w))
        self.assertEqual(imgs_hr.shape, (self.seq_len, 3, self.hr_h, self.hr_w))
        
        # We have seq_len-1 flow maps
        self.assertEqual(flows.shape, (self.seq_len - 1, 2, lr_h, lr_w))

        # Check tensor data types
        self.assertEqual(imgs_lr.dtype, torch.float32)
        self.assertEqual(imgs_hr.dtype, torch.float32)
        self.assertEqual(flows.dtype, torch.float32)

        # Check value ranges (images should be normalized to [0, 1])
        self.assertTrue(0.0 <= imgs_lr.min() and imgs_lr.max() <= 1.0)
        self.assertTrue(0.0 <= imgs_hr.min() and imgs_hr.max() <= 1.0)

if __name__ == '__main__':
    unittest.main()