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
        # Akışlar zaten düşük çözünürlükte olmalı, compute_flow'u LR görüntülerle çalıştırdıysanız.
        # Eğer HR görüntülerle çalıştırdıysanız, burada resize etmeniz gerekebilir.
        flows = torch.stack([torch.from_numpy(f.astype(np.float32)).permute(2, 0, 1) for f in flows])
        
        return {"imgs_lr": imgs_lr, "imgs_hr": imgs_hr, "flows": flows}