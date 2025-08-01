# tests/test_model.py
import unittest
import torch

# Test edilecek ana modelimizi import ediyoruz
from models.vsr_model import VSRModel

class TestVSRModel(unittest.TestCase):
    def test_forward_pass_shapes(self):
        """Modelin ileri besleme adımının doğru girdi ve çıktı şekilleriyle çalıştığını test eder."""
        # Test için parametreleri tanımla
        batch_size = 1  # Tek bir örnekle test ediyoruz
        channels = 3    # RGB
        lr_h, lr_w = 144, 256  # 144p (16:9 aspect ratio)
        hr_h, hr_w = 1080, 1920 # 1080p
        scale_factor = 7.5

        # Sahte (dummy) girdi tensörleri oluştur
        low_res_img = torch.randn(batch_size, channels, lr_h, lr_w)
        prev_high_res_img = torch.randn(batch_size, channels, hr_h, hr_w)
        flow = torch.randn(batch_size, 2, lr_h, lr_w) # Akış haritası 2 kanallıdır (x, y)

        # Modeli oluştur
        model = VSRModel(scale_factor=scale_factor)

        # Modeli çalıştır
        print("\nModel ileri besleme testi çalıştırılıyor...")
        output = model(low_res_img, prev_high_res_img, flow)
        print("Model başarıyla çalıştı.")

        # En önemli test: Çıktının şekli beklediğimiz 1080p şeklinde mi?
        expected_shape = (batch_size, channels, hr_h, hr_w)
        print(f"Beklenen Çıktı Şekli: {expected_shape}")
        print(f"Gerçek Çıktı Şekli:   {output.shape}")
        self.assertEqual(output.shape, expected_shape)

# Bu dosyayı doğrudan çalıştırmak için
if __name__ == '__main__':
    unittest.main()