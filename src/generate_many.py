# src/generate_many.py
import os, torch
from torchvision import utils
from src.model_generator import DCGANGenerator128
from PIL import Image
import torchvision.transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'
out_dir = 'inference/gen'
os.makedirs(out_dir, exist_ok=True)

# change to the checkpoint you want to evaluate
ckpt = 'results/G_epoch_50.pth'  

G = DCGANGenerator128().to(device)
G.load_state_dict(torch.load(ckpt, map_location=device))
G.eval()

# number of images to generate
N = 2000
batch = 64
z_dim = 100

idx = 0
with torch.no_grad():
    while idx < N:
        b = min(batch, N - idx)
        z = torch.randn(b, z_dim, device=device)
        imgs = G(z).cpu()  # range [-1,1]
        # convert to 0..1
        imgs = (imgs + 1) / 2
        for i in range(imgs.size(0)):
            img = imgs[i]
            # CHW -> HWC, to PIL
            grid = T.ToPILImage()(img)
            grid.save(os.path.join(out_dir, f'{idx+1:06d}.png'))
            idx += 1
print(f"Saved {N} images to {out_dir}")
