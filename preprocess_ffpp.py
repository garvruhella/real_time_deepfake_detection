import os
import cv2
import torch
from tqdm import tqdm
from facenet_pytorch import MTCNN
import shutil

# -----------------------------------------------
# 1️⃣ Configuration
# -----------------------------------------------
input_dir = "datasets/original_sequences/youtube/c23/videos"
output_dir = "datasets/FFPP_faces_original"
os.makedirs(output_dir, exist_ok=True)

# FORCE CPU ONLY (important)
device = 'cpu'
print(f"Using device: {device}")

# Initialize face detector on CPU only
mtcnn = MTCNN(keep_all=False, device=device)

# Get list of videos
video_list = sorted([v for v in os.listdir(input_dir) if v.endswith(".mp4")])
print(f"Found {len(video_list)} videos to process.")

# -----------------------------------------------
# 2️⃣ Process each video
# -----------------------------------------------
for name in tqdm(video_list):
    video_path = os.path.join(input_dir, name)
    video_name = os.path.splitext(name)[0]
    final_dir = os.path.join(output_dir, video_name)

    # Skip if already processed
    if os.path.exists(final_dir):
        tqdm.write(f"Skipping (already done): {video_name}")
        continue

    tmp_dir = os.path.join(output_dir, f"tmp_{video_name}")
    os.makedirs(tmp_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_i = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = mtcnn(frame_rgb)

            if face is not None:
                face_img = face.permute(1, 2, 0).int().cpu().numpy()
                face_img = face_img.astype('uint8')
                face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

                out_path = os.path.join(tmp_dir, f"{frame_i:06d}.jpg")
                cv2.imwrite(out_path, face_bgr)
                saved += 1

        except Exception as e:
            tqdm.write(f"⚠️ Frame {frame_i} failed in {video_name}: {e}")

        frame_i += 1

    cap.release()

    # If no faces were saved, remove temp folder
    if saved == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tqdm.write(f"⚠️ No faces found in {video_name}, skipped.")
        continue

    # Move temp folder to final
    try:
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        shutil.move(tmp_dir, final_dir)
        tqdm.write(f"✅ Finished {video_name} ({saved} faces saved).")
    except Exception as e:
        tqdm.write(f"⚠️ Could not move folder for {video_name}: {e}")

# -----------------------------------------------
# 3️⃣ Done
# -----------------------------------------------
print(" Face extraction complete.")