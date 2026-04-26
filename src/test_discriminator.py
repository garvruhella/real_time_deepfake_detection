import torch
from src.model_discriminator import DCGANDiscriminator128

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

D = DCGANDiscriminator128().to(device)
x = torch.randn(4, 3, 128, 128, device=device)  # 4 fake images
out = D(x)
print("✅ Output shape:", out.shape)
print("✅ Output values (real/fake probabilities):", out[:4].detach().cpu())
