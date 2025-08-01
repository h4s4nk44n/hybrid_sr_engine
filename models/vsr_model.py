# models/vsr_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Projemizin diğer modüllerinden gerekli fonksiyon ve sınıfları import ediyoruz
from .alignment import warp
from .sr_generator import RefinerCNN

class VSRModel(nn.Module):
    """
    Tüm temporal super-resolution boru hattını yöneten ana model (Generator).
    """
    def __init__(self, scale_factor=7.5): # 144p -> 1080p yaklaşık 7.5x
        super(VSRModel, self).__init__()
        # PyTorch'un interpolate fonksiyonu için scale_factor'ı sakla
        self.scale_factor = scale_factor
        
        # Hafif iyileştirici CNN'imizi oluşturuyoruz.
        # Girdisi 6 kanallı olacak: 
        #   - 3 kanal: Warp edilmiş geçmiş yüksek çözünürlüklü kare
        #   - 3 kanal: Mevcut düşük çözünürlüklü karenin kaba büyütülmüş hali
        self.refiner = RefinerCNN(in_channels=9, num_features=64, num_blocks=4)

    def forward(self, low_res_current: torch.Tensor, high_res_previous: torch.Tensor, flow_prev_to_current: torch.Tensor):
        """
        Modelin ana ileri besleme adımı.

        Args:
            low_res_current (torch.Tensor): Mevcut düşük çözünürlüklü kare (B, 3, 144, 256)
            high_res_previous (torch.Tensor): BİR ÖNCEKİ adımın ürettiği yüksek çözünürlüklü çıktı (B, 3, 1080, 1920)
            flow_prev_to_current (torch.Tensor): Düşük çözünürlükteki akış haritası (B, 2, 144, 256)
        """
        # Adım 1: Düşük çözünürlüklü akış haritasını hedef çözünürlüğe (1080p) büyüt.
        # Akış haritası için bilinear yeterince iyidir ve daha hızlıdır.
        flow_upscaled = F.interpolate(flow_prev_to_current, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        # Büyütme sonrası akış vektörlerinin büyüklüğünü de ölçekle çarpmak gerekir.
        flow_upscaled = flow_upscaled * self.scale_factor

        # Adım 2: Önceki 1080p kareyi, büyütülmüş akış haritasını kullanarak warp et.
        warped_history = warp(high_res_previous, flow_upscaled)

        # Adım 3: Mevcut 144p kareyi, bilinear enterpolasyon ile kaba bir şekilde 1080p'ye büyüt.
        upscaled_current = F.interpolate(low_res_current, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        
        # Adım 4: Warp edilmiş geçmişi (3 kanal) ve kaba büyütülmüş mevcut kareyi (3 kanal) birleştir.
        refiner_input = torch.cat([warped_history, upscaled_current], dim=1)
        
        # Adım 5: Birleştirilmiş tensörü, sadece "kusurları" düzeltmekle görevli RefinerCNN'e besle.
        # Modelden çıkan "artık" (residual), kaba görüntüdeki hataları düzeltmek için kullanılır.
        residual = self.refiner(refiner_input)
        
        # Adım 6: Kaba büyütülmüş görüntüye bu "düzeltmeyi" ekleyerek nihai, temiz 1080p görüntüyü oluştur.
        final_output = upscaled_current + residual
        
        # Çıktıyı [0, 1] aralığına sıkıştırarak (clamp) geçerli bir görüntü olduğundan emin olalım.
        final_output = torch.clamp(final_output, 0.0, 1.0)

        return final_output```