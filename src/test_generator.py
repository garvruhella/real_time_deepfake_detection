from src.model_generator import DCGANGenerator128
import torch

# Choose GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create generator and random latent noise
G = DCGANGenerator128().to(device)
z = torch.randn(4, 100, device=device)  # 4 samples of 100-dim noise

# Generate fake images
out = G(z)

print("✅ Output shape:", out.shape)
