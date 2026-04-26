# src/dataset.py
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class FFPPDataset(Dataset):
    """
    Loads images from datasets/training_data/<split>/{real,fake}/
    Returns (image_tensor, label) with images normalized to [-1,1].
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.samples = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
        split_dir = os.path.join(root_dir, split)
        # labels: real=0, fake=1
        for label, cls in enumerate(['real','fake']):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg','.png')):
                    self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label

def get_dataloader(root_dir, split='train', batch_size=32, num_workers=4, shuffle=True):
    ds = FFPPDataset(root_dir, split=split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
