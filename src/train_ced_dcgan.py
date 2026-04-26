import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.hybrid_common import IMAGE_SIZE, NORM_MEAN, NORM_STD
from src.model_generator import DCGANGenerator128
from src.model_discriminator import DCGANDiscriminator128


def main():
    data_root = "datasets/FFPP_faces"
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} for 50 epochs (mixed precision enabled)...")

    # --- Hyperparameters ---
    z_dim = 100
    batch_size = 32
    epochs = 50
    lr = 0.0002
    beta1 = 0.5

    # --- Data ---
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # --- Models ---
    G = DCGANGenerator128().to(device)
    D = DCGANDiscriminator128().to(device)
    print(next(G.parameters()).device)

    # ✅ Use BCEWithLogitsLoss (safe for AMP)
    criterion = nn.BCEWithLogitsLoss()

    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    scaler = GradScaler()

    fixed_noise = torch.randn(16, z_dim, device=device)

    start_epoch = 0
    ckpts = [f for f in os.listdir("results") if f.startswith("G_epoch_")]
    if ckpts:
        last_epoch = max(int(f.split("_")[2].split(".")[0]) for f in ckpts)
        print(f"Resuming from epoch {last_epoch}...")
        G.load_state_dict(torch.load(f"results/G_epoch_{last_epoch}.pth"))
        D.load_state_dict(torch.load(f"results/D_epoch_{last_epoch}.pth"))
        start_epoch = last_epoch

    # --- Training Loop ---
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        for i, (real_images, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)):
            batch_size_now = real_images.size(0)
            real_images = real_images.to(device, non_blocking=True)
            real_label = torch.ones(batch_size_now, 1, device=device)
            fake_label = torch.zeros(batch_size_now, 1, device=device)

            # ---- Train Discriminator ----
            optimizerD.zero_grad()
            noise = torch.randn(batch_size_now, z_dim, device=device)
            with autocast():
                output_real = D(real_images)
                output_real = output_real.view(batch_size_now, -1).mean(1, keepdim=True)
                lossD_real = criterion(output_real, real_label)

                fake_images = G(noise)
                output_fake = D(fake_images.detach())
                output_fake = output_fake.view(batch_size_now, -1).mean(1, keepdim=True)
                lossD_fake = criterion(output_fake, fake_label)

                lossD = lossD_real + lossD_fake

            scaler.scale(lossD).backward()
            scaler.step(optimizerD)
            scaler.update()

            # ---- Train Generator ----
            optimizerG.zero_grad()
            with autocast():
                output_fake_for_G = D(fake_images)
                output_fake_for_G = output_fake_for_G.view(batch_size_now, -1).mean(1, keepdim=True)
                lossG = criterion(output_fake_for_G, real_label)

            scaler.scale(lossG).backward()
            scaler.step(optimizerG)
            scaler.update()

            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(loader)}] "
                      f"Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")

        with torch.no_grad():
            fake = G(fixed_noise).detach().cpu()
            utils.save_image(fake, f"results/fake_epoch_{epoch+1}.png", normalize=True, nrow=4)

        torch.save(G.state_dict(), f"results/G_epoch_{epoch+1}.pth")
        torch.save(D.state_dict(), f"results/D_epoch_{epoch+1}.pth")
        print(f"✅ Epoch {epoch+1} done in {(time.time()-epoch_start)/60:.1f} min")


if __name__ == "__main__":
    main()
