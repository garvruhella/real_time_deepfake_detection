import os
import hashlib
from glob import glob

def hash_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

train_images = glob("datasets/final_data/train/*/*.jpg")
val_images = glob("datasets/final_data/val/*/*.jpg")

print(f"Train images: {len(train_images)}")
print(f"Val images: {len(val_images)}")

print("Hashing train images...")
train_hashes = set(hash_file(p) for p in train_images)

print("Hashing val images...")
val_hashes = set(hash_file(p) for p in val_images)

overlap = train_hashes.intersection(val_hashes)

print(f"Duplicate images across train & val: {len(overlap)}")