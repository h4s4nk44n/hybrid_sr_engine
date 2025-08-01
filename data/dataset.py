import os, cv2, numpy as np
import torch
from torch.utils.data import Dataset

class TemporalFrameDataset(Dataset):
    def __init__(self, root_dir, seq_len=4, transform=None):
        self.root = root_dir
        self.seq_len = seq_len
        self.transform = transform
        self.samples = []
        for vid in sorted(os.listdir(root_dir)):
            vidpath = os.path.join(root_dir, vid)
            frames = sorted([f for f in os.listdir(vidpath) if f.endswith((".png",".jpg", ".jpeg"))])
            for idx in range(seq_len - 1, len(frames)):
                self.samples.append((vidpath, frames, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vidpath, frames, frame_idx = self.samples[idx]
        imgs, flows = [], []
        for i in range(frame_idx - self.seq_len + 1, frame_idx + 1):
            img = cv2.cvtColor(cv2.imread(os.path.join(vidpath, frames[i])), cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        for i in range(frame_idx - self.seq_len + 2, frame_idx + 1):
            fname = os.path.join(vidpath, f"flow_{i-1:04d}_{i:04d}.npy")
            flows.append(np.load(fname))
        imgs = torch.stack([torch.from_numpy(im).permute(2,0,1).float()/255 for im in imgs])
        flows = torch.stack([torch.from_numpy(f).permute(2,0,1).float() for f in flows])
        return {"imgs": imgs, "flows": flows}
