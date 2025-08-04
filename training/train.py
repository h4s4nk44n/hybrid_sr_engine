# training/train.py (Global Iteration Counter ile güncellenmiş versiyon)
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.utils as vutils

# Projemizin diğer dosyalarından gerekli sınıfları import ediyoruz
from data.dataset import TemporalFrameDataset
from models.vsr_model import VSRModel
from models.discriminator import Discriminator
from training.losses import CombinedLoss
from training.metrics import psnr # Metrikleri de import edelim

# ==============================================================================
# GÖRSEL DOĞRULAMA (İterasyon numarası ile güncellendi)
# ==============================================================================
def validate_and_save_image(generator, val_loader, iteration, device, output_dir, scale_factor):
    """Her belirli iterasyon sonunda bir validation örneği üzerinde modeli çalıştırır ve sonucu kaydeder."""
    generator.eval()
    with torch.no_grad():
        try:
            val_batch = next(iter(val_loader))
        except StopIteration:
            generator.train()
            return

        low_res_seq = val_batch['imgs_lr'].to(device)
        high_res_seq = val_batch['imgs_hr'].to(device)
        flow_seq = val_batch['flows'].to(device)
        high_res_previous = torch.zeros_like(high_res_seq[:, 0]).to(device)

        for t in range(low_res_seq.size(1)):
            low_res_current = low_res_seq[:, t]
            flow_prev_to_current = flow_seq[:, t-1] if t > 0 else torch.zeros(low_res_current.size(0), 2, low_res_current.shape[2], low_res_current.shape[3]).to(device)
            fake_high_res = generator(low_res_current, high_res_previous, flow_prev_to_current)
            high_res_previous = fake_high_res

        model_output = fake_high_res
        ground_truth = high_res_seq[:, -1]
        current_psnr = psnr(model_output, ground_truth)
        
        baseline = torch.nn.functional.interpolate(low_res_current, scale_factor=scale_factor, mode='bilinear')
        comparison_grid = torch.cat([baseline[0].cpu(), model_output[0].cpu(), ground_truth[0].cpu()], dim=2)
        
        # YENİ: Dosya adında iterasyon numarasını kullan
        save_path = os.path.join(output_dir, f"validation_step_{iteration}.png")
        vutils.save_image(comparison_grid, save_path, normalize=False)
        print(f"\nValidation Step {iteration} | PSNR: {current_psnr.item():.2f} dB | Resim kaydedildi: {save_path}")

    generator.train()

# ==============================================================================
# ANA EĞİTİM FONKSİYONU (İterasyon sayacı ile güncellendi)
# ==============================================================================
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")

    # --- Veri Yükleyicileri ---
    train_dataset = TemporalFrameDataset(root_dir=args.data_dir, seq_len=args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = TemporalFrameDataset(root_dir=args.val_dir, seq_len=args.seq_len)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --- Modeller, Kayıplar, Optimizer'lar ---
    generator = VSRModel(scale_factor=7.5).to(device)
    discriminator = Discriminator().to(device)
    loss_content = CombinedLoss(l1_weight=1.0, perceptual_weight=0.1).to(device)
    loss_gan = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer_g = optim.AdamW(generator.parameters(), lr=args.lr_g, betas=(0.9, 0.999))
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=args.lr_d, betas=(0.9, 0.999))

    # YENİ: Klasörleri ve global iterasyon sayacını oluştur
    val_image_dir = os.path.join(args.output_dir, 'validation_images')
    checkpoint_dir = os.path.join(args.output_dir, 'model_checkpoints')
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    global_step = 0
    print("GAN ile eğitim başlıyor...")

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        # Epoch ilerlemesini göstermek için `tqdm` kullanmaya devam ediyoruz
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in progress_bar:
            global_step += 1 # Her batch bir iterasyondur
            
            low_res_seq = batch['imgs_lr'].to(device)
            high_res_seq = batch['imgs_hr'].to(device)
            flow_seq = batch['flows'].to(device)
            high_res_previous = torch.zeros_like(high_res_seq[:, 0]).to(device)
            
            # (İç temporal döngü aynı kaldı)
            for t in range(args.seq_len):
                low_res_current = low_res_seq[:, t]
                high_res_target = high_res_seq[:, t]
                flow_prev_to_current = flow_seq[:, t-1] if t > 0 else torch.zeros(args.batch_size, 2, low_res_current.shape[2], low_res_current.shape[3]).to(device)

                # >>> Generator Eğitimi <<<
                optimizer_g.zero_grad()
                fake_high_res = generator(low_res_current, high_res_previous.detach(), flow_prev_to_current)
                pred_fake = discriminator(fake_high_res)
                gan_loss_g = loss_gan(pred_fake, torch.ones_like(pred_fake))
                content_loss = loss_content(fake_high_res, high_res_target)
                total_loss_g = content_loss + args.gan_weight * gan_loss_g
                total_loss_g.backward()
                optimizer_g.step()

                # >>> Discriminator Eğitimi <<<
                optimizer_d.zero_grad()
                pred_real = discriminator(high_res_target)
                loss_real = loss_gan(pred_real, torch.ones_like(pred_real))
                pred_fake = discriminator(fake_high_res.detach())
                loss_fake = loss_gan(pred_fake, torch.zeros_like(pred_fake))
                total_loss_d = (loss_real + loss_fake) / 2
                total_loss_d.backward()
                optimizer_d.step()
                
                high_res_previous = fake_high_res.detach()
            
            # YENİ: İlerleme çubuğuna global adımı da ekle
            progress_bar.set_postfix({
                'Step': global_step,
                'G_loss': f"{total_loss_g.item():.4f}", 
                'D_loss': f"{total_loss_d.item():.4f}"
            })

            # === DÖNGÜ İÇİ EYLEMLER ===
            # Belirli iterasyon aralıklarında validation yap ve kaydet
            if global_step % args.val_interval == 0:
                validate_and_save_image(generator, val_loader, global_step, device, val_image_dir, 7.5)

            # Belirli iterasyon aralıklarında checkpoint kaydet
            if global_step % args.save_interval == 0:
                torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f'generator_step_{global_step}.pth'))
                torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f'discriminator_step_{global_step}.pth'))

    print("Eğitim tamamlandı!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VSR-GAN Model Eğitim Betiği (İterasyon Sayacı ile)')
    parser.add_argument('--data_dir', type=str, required=True, help='Eğitim veri seti kök dizini.')
    parser.add_argument('--val_dir', type=str, required=True, help='Validation veri seti kök dizini.')
    parser.add_argument('--output_dir', type=str, default='training_results', help='Tüm çıktıların (checkpoint, resim) kaydedileceği ana dizin.')
    parser.add_argument('--epochs', type=int, default=100, help='Toplam epoch sayısı.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch boyutu.')
    parser.add_argument('--lr_g', type=float, default=1e-4, help='Generator öğrenme oranı.')
    parser.add_argument('--lr_d', type=float, default=4e-4, help='Discriminator öğrenme oranı.')
    parser.add_argument('--seq_len', type=int, default=8, help='Ardışık kare sayısı.')
    parser.add_argument('--gan_weight', type=float, default=0.01, help='Generator kaybındaki GAN kaybının ağırlığı.')
    
    # YENİ: İterasyon bazlı kaydetme ve doğrulama aralıkları
    parser.add_argument('--save_interval', type=int, default=5000, help='Kaç iterasyonda bir checkpoint kaydedileceği.')
    parser.add_argument('--val_interval', type=int, default=1000, help='Kaç iterasyonda bir validation resmi kaydedileceği.')
    
    args = parser.parse_args()
    train(args)