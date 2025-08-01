# training/train.py (Görsel Doğrulama ile güncellenmiş versiyon)
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.utils as vutils # Görüntüleri kaydetmek için YENİ import

# Projemizin diğer dosyalarından gerekli sınıfları import ediyoruz
from data.dataset import TemporalFrameDataset
from models.vsr_model import VSRModel
from models.discriminator import Discriminator
from training.losses import CombinedLoss

# ==============================================================================
# YENİ FONKSİYON: Görsel Doğrulama
# ==============================================================================
def validate_and_save_image(generator, val_loader, epoch, device, output_dir, scale_factor):
    """Her epoch sonunda bir validation örneği üzerinde modeli çalıştırır ve sonucu kaydeder."""
    generator.eval() # Modeli değerlendirme moduna al
    with torch.no_grad(): # Gradient hesaplamayı devre dışı bırak
        # Validation setinden sadece bir batch al
        try:
            val_batch = next(iter(val_loader))
        except StopIteration:
            print("Validation loader boş.")
            generator.train()
            return

        # Veriyi cihaza gönder
        low_res_seq = val_batch['imgs_lr'].to(device)
        high_res_seq = val_batch['imgs_hr'].to(device)
        flow_seq = val_batch['flows'].to(device)

        # Geçmişi sıfırla
        high_res_previous = torch.zeros_like(high_res_seq[:, 0]).to(device)

        # Dizi boyunca ilerle
        for t in range(low_res_seq.size(1)):
            low_res_current = low_res_seq[:, t]
            flow_prev_to_current = flow_seq[:, t-1] if t > 0 else torch.zeros(low_res_current.size(0), 2, low_res_current.shape[2], low_res_current.shape[3]).to(device)
            
            # Son çıktıyı al
            fake_high_res = generator(low_res_current, high_res_previous, flow_prev_to_current)
            high_res_previous = fake_high_res # detach'a gerek yok, no_grad() içindeyiz

        # Karşılaştırma için dizinin son karesini kullan
        # Baseline: Basit bilinear büyütme
        baseline = torch.nn.functional.interpolate(low_res_current, scale_factor=scale_factor, mode='bilinear')
        # Modelin çıktısı
        model_output = fake_high_res
        # Gerçek Görüntü
        ground_truth = high_res_seq[:, -1]

        # Görüntüleri bir grid'de birleştir (Sırasıyla: Baseline, Bizim Model, Gerçek Görüntü)
        # Sadece batch'in ilk örneğini kaydet
        comparison_grid = torch.cat([baseline[0].cpu(), model_output[0].cpu(), ground_truth[0].cpu()], dim=2)
        
        # Grid'i kaydet
        save_path = os.path.join(output_dir, f"validation_epoch_{epoch+1}.png")
        vutils.save_image(comparison_grid, save_path, normalize=False)

    generator.train() # Modeli tekrar eğitim moduna al
    print(f"Validation resmi kaydedildi: {save_path}")


# ==============================================================================
# ANA EĞİTİM FONKSİYONU (Güncellenmiş)
# ==============================================================================
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")

    # --- Veri Yükleyicileri (YENİ: Train ve Val için ayrı) ---
    train_dataset = TemporalFrameDataset(root_dir=args.data_dir, seq_len=args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    val_dataset = TemporalFrameDataset(root_dir=args.val_dir, seq_len=args.seq_len)
    # Validation için shuffle=False ve batch_size=1 genellikle daha iyidir.
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --- Modeller, Kayıplar, Optimizer'lar ---
    generator = VSRModel(scale_factor=7.5).to(device)
    discriminator = Discriminator().to(device)
    loss_content = CombinedLoss(l1_weight=1.0, perceptual_weight=0.1).to(device)
    loss_gan = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer_g = optim.AdamW(generator.parameters(), lr=args.lr_g, betas=(0.9, 0.999))
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=args.lr_d, betas=(0.9, 0.999))

    os.makedirs(args.output_dir, exist_ok=True)
    print("GAN ile eğitim başlıyor...")

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in progress_bar:
            # (Eğitim döngüsü aynı kaldı, burası değişmedi)
            low_res_seq = batch['imgs_lr'].to(device)
            high_res_seq = batch['imgs_hr'].to(device)
            flow_seq = batch['flows'].to(device)
            high_res_previous = torch.zeros_like(high_res_seq[:, 0]).to(device)
            
            for t in range(args.seq_len):
                low_res_current = low_res_seq[:, t]
                high_res_target = high_res_seq[:, t]
                flow_prev_to_current = flow_seq[:, t-1] if t > 0 else torch.zeros(args.batch_size, 2, low_res_current.shape[2], low_res_current.shape[3]).to(device)

                # >>> Adım 1: Generator'ı Eğit <<<
                optimizer_g.zero_grad()
                fake_high_res = generator(low_res_current, high_res_previous.detach(), flow_prev_to_current)
                pred_fake = discriminator(fake_high_res)
                gan_loss_g = loss_gan(pred_fake, torch.ones_like(pred_fake))
                content_loss = loss_content(fake_high_res, high_res_target)
                total_loss_g = content_loss + args.gan_weight * gan_loss_g
                total_loss_g.backward()
                optimizer_g.step()

                # >>> Adım 2: Discriminator'ı Eğit <<<
                optimizer_d.zero_grad()
                pred_real = discriminator(high_res_target)
                loss_real = loss_gan(pred_real, torch.ones_like(pred_real))
                pred_fake = discriminator(fake_high_res.detach())
                loss_fake = loss_gan(pred_fake, torch.zeros_like(pred_fake))
                total_loss_d = (loss_real + loss_fake) / 2
                total_loss_d.backward()
                optimizer_d.step()
                
                high_res_previous = fake_high_res.detach()
            
            progress_bar.set_postfix({'G_loss': total_loss_g.item(), 'D_loss': total_loss_d.item()})

        # === DÖNGÜ SONU EYLEMLERİ ===
        # Her epoch sonunda validation yap ve resmi kaydet
        validate_and_save_image(generator, val_loader, epoch, device, args.output_dir, 7.5)

        # Checkpoint kaydet
        if (epoch + 1) % args.save_interval == 0:
            torch.save(generator.state_dict(), os.path.join(args.output_dir, f'generator_epoch_{epoch+1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(args.output_dir, f'discriminator_epoch_{epoch+1}.pth'))

    print("Eğitim tamamlandı!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VSR-GAN Model Eğitim Betiği (Validation ile)')
    # YENİ: Validation veri seti için argüman
    parser.add_argument('--data_dir', type=str, required=True, help='Eğitim için işlenmiş veri kök dizini.')
    parser.add_argument('--val_dir', type=str, required=True, help='Validation için işlenmiş veri kök dizini.')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Checkpoint ve validation resimlerinin kayıt dizini.')
    parser.add_argument('--epochs', type=int, default=200, help='Toplam epoch sayısı.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch boyutu.')
    parser.add_argument('--lr_g', type=float, default=1e-4, help='Generator öğrenme oranı.')
    parser.add_argument('--lr_d', type=float, default=4e-4, help='Discriminator öğrenme oranı.')
    parser.add_argument('--seq_len', type=int, default=8, help='Ardışık kare sayısı.')
    parser.add_argument('--gan_weight', type=float, default=0.01, help='Generator kaybındaki GAN kaybının ağırlığı.')
    parser.add_argument('--save_interval', type=int, default=5, help='Kaç epoch\'ta bir checkpoint kaydedileceği.')
    
    args = parser.parse_args()
    train(args)