# prepare_training_data.py (fixed: keeps video_id folders)
#
# Splits by VIDEO id (not frame): train/val are disjoint at the video level.
# After building, run:  python check_video_overlap.py
# And for duplicate pixels across splits:  python check_leakage.py

import os, random, shutil
from glob import glob
from tqdm import tqdm

FAKE_DIR = "datasets/FFPP_faces"
REAL_DIR = "datasets/FFPP_faces_original"
OUT = "datasets/final_data"

MAX_PER_VIDEO = 10
FRAME_STEP = 15
TRAIN_SPLIT = 0.8
random.seed(42)

def sample_frames(video_dir):
    imgs = sorted(glob(os.path.join(video_dir, "*.jpg")))
    imgs = imgs[::FRAME_STEP]
    if len(imgs) > MAX_PER_VIDEO:
        imgs = random.sample(imgs, MAX_PER_VIDEO)
    return imgs

def split(videos):
    random.shuffle(videos)
    k = int(len(videos)*TRAIN_SPLIT)
    return videos[:k], videos[k:]

fake_videos = [v for v in os.listdir(FAKE_DIR) if os.path.isdir(os.path.join(FAKE_DIR, v))]
real_videos = [v for v in os.listdir(REAL_DIR) if os.path.isdir(os.path.join(REAL_DIR, v))]

f_tr, f_va = split(fake_videos)
r_tr, r_va = split(real_videos)

def copy_set(videos, base, split_name, cls):
    for vid in tqdm(videos, desc=f"{split_name}-{cls}"):
        src = os.path.join(base, vid)
        dst = os.path.join(OUT, split_name, cls, vid)
        os.makedirs(dst, exist_ok=True)
        frames = sample_frames(src)
        for i, p in enumerate(frames):
            shutil.copy(p, os.path.join(dst, f"{i}.jpg"))

# create dirs
for s in ["train","val"]:
    for c in ["fake","real"]:
        os.makedirs(os.path.join(OUT, s, c), exist_ok=True)

copy_set(f_tr, FAKE_DIR, "train", "fake")
copy_set(r_tr, REAL_DIR, "train", "real")
copy_set(f_va, FAKE_DIR, "val",   "fake")
copy_set(r_va, REAL_DIR, "val",   "real")

print("✅ Dataset rebuilt with video-level split")