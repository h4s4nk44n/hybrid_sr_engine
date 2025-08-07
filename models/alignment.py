# models/alignment.py
import torch
import torch.nn.functional as F

def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Bir görüntüyü (veya bir grup görüntüyü) optik akış vektörlerine göre hizalar (warp).

    Args:
        image (torch.Tensor): Hizalanacak görüntü. Shape: (B, C, H, W)
        flow (torch.Tensor): Görüntüyü nasıl hizalayacağını belirten akış haritası. Shape: (B, 2, H, W)

    Returns:
        torch.Tensor: Hizalanmış görüntü. Shape: (B, C, H, W)
    """
    B, C, H, W = image.size()

    # grid_sample için bir koordinat grid'i oluştur (normalleştirilmiş: -1 ile 1 arası)
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    grid = torch.stack([xx, yy], dim=0).float()
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    # Grid'i görüntünün bulunduğu cihaza gönder
    grid = grid.to(image.device)

    # Akış (flow) piksellerin NEREYE GİTTİĞİNİ gösterir (göreli hareket).
    # new_grid ise her pikselin NEREDEN GELECEĞİNİ gösteren mutlak koordinatları içermelidir.
    # Bu yüzden mevcut grid koordinatlarına akış vektörlerini ekliyoruz.
    new_grid = grid + flow

    # Koordinatları grid_sample'ın beklediği [-1, 1] aralığına normalleştir
    new_grid[:, 0, :, :] = 2 * new_grid[:, 0, :, :] / (W - 1) - 1
    new_grid[:, 1, :, :] = 2 * new_grid[:, 1, :, :] / (H - 1) - 1

    # grid_sample (B, H, W, 2) formatını beklediği için boyutları permute et
    new_grid = new_grid.permute(0, 2, 3, 1)

    # Warp işlemini gerçekleştir
    warped_image = F.grid_sample(image, new_grid, mode='bilinear', padding_mode='border', align_corners=True)

    return warped_image