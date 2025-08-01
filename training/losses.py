# training/losses.py
import torch
import torch.nn as nn
import lpips

class CombinedLoss(nn.Module):
    """
    L1 ve LPIPS kayıplarını birleştiren ve ağırlıklandıran birleşik kayıp fonksiyonu.
    """
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
        # Temel kayıp fonksiyonlarını oluştur
        self.l1_loss = nn.L1Loss()
        
        # LPIPS modelini yükle. 'alex' ağı genellikle yeterlidir ve hızlıdır.
        # Bu model, ilk kullanımda otomatik olarak indirilecektir.
        self.perceptual_loss = lpips.LPIPS(net='alex')

    def forward(self, predicted, target):
        """
        Tahmin edilen ve hedef görüntüler arasındaki toplam kaybı hesaplar.

        Args:
            predicted (torch.Tensor): Modelin ürettiği yüksek çözünürlüklü görüntü.
            target (torch.Tensor): Gerçek (ground truth) yüksek çözünürlüklü görüntü.

        Returns:
            torch.Tensor: Toplam, ağırlıklandırılmış kayıp değeri.
        """
        # L1 kaybını hesapla
        loss_l1 = self.l1_loss(predicted, target)
        
        # LPIPS kaybını hesapla
        # LPIPS, [-1, 1] aralığında normalleştirilmiş görüntüler bekler, biz ise [0, 1] aralığındayız.
        # Bu yüzden 2*x - 1 dönüşümünü uyguluyoruz.
        loss_lpips = self.perceptual_loss(predicted * 2 - 1, target * 2 - 1).mean()
        
        # Kayıpları ağırlıklarıyla birleştir
        total_loss = (self.l1_weight * loss_l1) + (self.perceptual_weight * loss_lpips)
        
        return total_loss