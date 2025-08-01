import torch
import torch.nn.functional as F

def psnr(predicted: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    İki görüntü arasındaki Peak Signal-to-Noise Ratio (PSNR) değerini hesaplar.
    Görüntülerin [0, max_val] aralığında olması beklenir.
    """
    # Ortalama Kare Hatasını (Mean Squared Error) hesapla
    mse = F.mse_loss(predicted, target)
    
    # MSE sıfırsa (görüntüler aynıysa), PSNR sonsuzdur. Yüksek bir değer döndür.
    if mse == 0:
        return torch.tensor(float('inf'))
    
    # PSNR formülünü uygula
    psnr_val = 20 * torch.log10(max_val / torch.sqrt(mse))
    
    return psnr_val