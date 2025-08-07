# models/sr_generator.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Basit bir residual blok."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual # Artık bağlantısı (Skip connection)
        return out

class RefinerCNN(nn.Module):
    """
    Yüksek çözünürlüklü bir girdiyi alıp, detayları iyileştiren çok hafif bir CNN.
    Bu model, upscaling YAPMAZ. Girdi ve çıktı çözünürlüğü aynıdır.
    """
    def __init__(self, in_channels=6, num_features=64, num_blocks=4):
        super(RefinerCNN, self).__init__()
        # İlk katman: Girdi kanallarını (örn. 6) modelin çalışma kanallarına (örn. 64) çıkarır
        self.input_conv = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        # Artık bloklar
        blocks = [ResidualBlock(num_features) for _ in range(num_blocks)]
        self.residual_layers = nn.Sequential(*blocks)

        # Çıktı katmanı: Modelin kanallarını (64) tekrar resim kanallarına (3) indirir
        self.output_conv = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.residual_layers(x)
        x = self.output_conv(x)
        return x