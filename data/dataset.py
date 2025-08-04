import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class TemporalFrameDataset(Dataset):
    def __init__(self, root_dir, seq_len=8, lr_height=144, transform=None):
        self.root = root_dir
        self.seq_len = seq_len
        self.transform = transform
        
        # Düşük çözünürlük boyutlarını sakla
        self.lr_h = lr_height
        # 16:9 en-boy oranını koruyarak genişliği hesapla
        self.lr_w = int(lr_height * 16.0 / 9.0)
        
        self.samples = []
        # Veri setindeki videoları (veya frame klasörlerini) bul
        for vid_folder in sorted(os.listdir(root_dir)):
            vidpath = os.path.join(root_dir, vid_folder)
            if not os.path.isdir(vidpath):
                continue
            
            frames = sorted([f for f in os.listdir(vidpath) if f.endswith((".png", ".jpg", ".jpeg"))])
            
            # Yeterli uzunlukta diziler (sequence) oluştur
            if len(frames) >= seq_len:
                for idx in range(len(frames) - seq_len + 1):
                    # Bir sequence, başlangıç indeksi ve klasör yolu ile temsil edilir
                    self.samples.append((vidpath, frames, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vidpath, frames, start_idx = self.samples[idx]
        
        imgs_hr, imgs_lr, flows = [], [], []
        
        # Sequence boyunca kareleri ve akışları yükle
        for i in range(start_idx, start_idx + self.seq_len):
            # --- Yüksek Çözünürlüklü Kare ---
            hr_path = os.path.join(vidpath, frames[i])
            img_hr = cv2.imread(hr_path)
            img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB) # RGB'ye çevir
            
            # --- Düşük Çözünürlüklü Kare (YENİ) ---
            # Yüksek çözünürlüklü kareden anlık olarak oluştur
            # Görüntüyü küçültürken INTER_AREA enterpolasyonu en iyi sonucu verir.
            img_lr = cv2.resize(img_hr, (self.lr_w, self.lr_h), interpolation=cv2.INTER_AREA)

            # Opsiyonel transformları uygula (örn. data augmentation)
            if self.transform:
                img_hr = self.transform(img_hr)
                img_lr = self.transform(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)
            
            # --- Optik Akış ---
            # İlk kare hariç diğerleri için akışı yükle
            if i > start_idx:
                flow_path = os.path.join(vidpath, f"flow_{i-1:04d}_{i:04d}.npy")
                # Dosya adlandırmanız farklıysa burayı güncelleyin (örn: flow_{frames[i-1]}_{frames[i]}.npy)
                flow = np.load(flow_path)
                flows.append(flow)
        
        # İlk kare için sahte bir akış ekle (kullanılmayacak ama tensör boyutları tutarlı olmalı)
        # Not: train.py'de bu zaten ele alınıyor ama burada da boyut tutarlılığı için ekleyebiliriz.
        # Bu satır yerine train.py'deki mantık daha güvenilir. Şimdilik listede seq_len-1 akış var.

        # Görüntüleri [0, 1] aralığına normalize et ve (C, H, W) formatına çevir
        imgs_hr = torch.stack([torch.from_numpy(im.astype(np.float32)).permute(2, 0, 1) / 255.0 for im in imgs_hr])
        imgs_lr = torch.stack([torch.from_numpy(im.astype(np.float32)).permute(2, 0, 1) / 255.0 for im in imgs_lr])
        
        # Akışları (C, H, W) formatına çevir
        processed_flows = []
        for flow_hr in flows:
            # The original flow was calculated on HR frames. We must downscale it
            # to the LR resolution to simulate the inference-time condition.
            B, H, W, C = 1, self.lr_h, self.lr_w, 2
            flow_lr = cv2.resize(flow_hr, (W, H), interpolation=cv2.INTER_AREA)
            
            # When you resize a flow map, you must also scale the magnitude of the vectors.
            # Original flow vectors were for a 1080p grid, now they are for a 144p grid.
            hr_h, hr_w, _ = img_hr.shape # Get the actual HR shape
            scale_h = H / hr_h
            scale_w = W / hr_w
            flow_lr[:, :, 0] = flow_lr[:, :, 0] * scale_w
            flow_lr[:, :, 1] = flow_lr[:, :, 1] * scale_h
            
            processed_flows.append(torch.from_numpy(flow_lr.astype(np.float32)).permute(2, 0, 1))
            
        flows = torch.stack(processed_flows)
        
        return {"imgs_lr": imgs_lr, "imgs_hr": imgs_hr, "flows": flows}