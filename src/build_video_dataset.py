import os
import shutil
import random
import argparse

def build_dataset(faces_root, out_root, train_ratio=0.8):
    os.makedirs(out_root, exist_ok=True)
    for split in ['train', 'test']:
        for cls in ['real', 'fake']:
            os.makedirs(os.path.join(out_root, split, cls), exist_ok=True)

    for label in ['real', 'fake']:
        label_dir = os.path.join(faces_root, label + "1")  # example: real1, fake1
        if not os.path.exists(label_dir):
            # try generic folder names like "real", "fake"
            label_dir = os.path.join(faces_root, label)
        if not os.path.exists(label_dir):
            print(f"⚠️ Missing folder for {label}")
            continue

        images = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.png')]
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_imgs, test_imgs = images[:split_idx], images[split_idx:]

        print(f"📸 {label.upper()}: {len(images)} total → {len(train_imgs)} train, {len(test_imgs)} test")

        for img in train_imgs:
            shutil.copy(img, os.path.join(out_root, 'train', label, os.path.basename(img)))
        for img in test_imgs:
            shutil.copy(img, os.path.join(out_root, 'test', label, os.path.basename(img)))

    print("\n✅ Detector dataset built successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--faces_root", required=True, help="Path to extracted faces")
    parser.add_argument("--out_root", required=True, help="Output dataset root")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/test split ratio")
    args = parser.parse_args()

    build_dataset(args.faces_root, args.out_root, train_ratio=args.train_ratio)
