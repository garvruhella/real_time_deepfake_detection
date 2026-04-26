import os
import cv2
import random
from tqdm import tqdm

# =========================
# CONFIG
# =========================
REAL_VIDEOS_DIR = "datasets/original_sequences/youtube/c23/videos"
FAKE_VIDEOS_DIR = "datasets/manipulated_sequences/Deepfakes/c23/videos"

OUTPUT_DIR = "datasets/final_data"

NUM_REAL = 50
NUM_FAKE = 50
TRAIN_SPLIT = 0.8

MAX_FRAMES_PER_VIDEO = 5
FRAME_STEP = 25  # skip frames (reduces duplicates)

IMG_SIZE = 128

random.seed(42)

# =========================
# HELPERS
# =========================
def extract_frames(video_path, save_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    saved = 0

    while cap.isOpened() and saved < MAX_FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break

        if count % FRAME_STEP == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            filename = os.path.join(
                save_dir,
                f"{os.path.basename(video_path).replace('.mp4','')}_{saved}.jpg"
            )
            cv2.imwrite(filename, frame)
            saved += 1

        count += 1

    cap.release()


def process_split(video_list, label, split):
    save_dir = os.path.join(OUTPUT_DIR, split, label)
    os.makedirs(save_dir, exist_ok=True)

    for video in tqdm(video_list, desc=f"{split}/{label}"):
        extract_frames(video, save_dir)


# =========================
# LOAD VIDEOS
# =========================
real_videos = [
    os.path.join(REAL_VIDEOS_DIR, v)
    for v in os.listdir(REAL_VIDEOS_DIR)
    if v.endswith(".mp4")
]

fake_videos = [
    os.path.join(FAKE_VIDEOS_DIR, v)
    for v in os.listdir(FAKE_VIDEOS_DIR)
    if v.endswith(".mp4")
]

# shuffle
random.shuffle(real_videos)
random.shuffle(fake_videos)

# take subset
real_videos = real_videos[:NUM_REAL]
fake_videos = fake_videos[:NUM_FAKE]

# =========================
# SPLIT (VIDEO LEVEL)
# =========================
def split_list(lst):
    split_idx = int(len(lst) * TRAIN_SPLIT)
    return lst[:split_idx], lst[split_idx:]

real_train, real_val = split_list(real_videos)
fake_train, fake_val = split_list(fake_videos)

# =========================
# CLEAR OLD DATA
# =========================
import shutil
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# =========================
# PROCESS
# =========================
print("Processing REAL videos...")
process_split(real_train, "real", "train")
process_split(real_val, "real", "val")

print("Processing FAKE videos...")
process_split(fake_train, "fake", "train")
process_split(fake_val, "fake", "val")

print("✅ Dataset created successfully!")