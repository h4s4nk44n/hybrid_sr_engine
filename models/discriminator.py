import torch.nn as nn

class Discriminator(nn.Module):
    """
    PatchGAN tabanlı basit bir ayrımcı model.
    Bir görüntünün gerçek mi yoksa sahte mi olduğunu sınıflandırır.
    Girdi: (B, 3, H, W) -> Çıktı: (B, 1, H/8, W/8) - Her bir patch için bir "gerçeklik" skoru.
    """
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            """Bir ayrımcı bloğu döndürür."""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, padding=1) # Son katman, gerçeklik haritasını üretir.
        )

    def forward(self, img):
        return self.model(img)