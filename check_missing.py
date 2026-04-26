import os

video_dir = r"datasets\manipulated_sequences\Deepfakes\c23\videos"
face_dir  = r"datasets\FFPP_faces"

# All video file names (without .mp4)
videos = [os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.endswith(".mp4")]
# All processed folder names
processed = [d for d in os.listdir(face_dir) if os.path.isdir(os.path.join(face_dir, d))]

# Find what’s missing
missing = sorted(set(videos) - set(processed))

print(f"Total videos: {len(videos)}")
print(f"Processed folders: {len(processed)}")
print(f"Missing count: {len(missing)}")

if missing:
    print("\nMissing examples:")
    for m in missing[:20]:
        print(" -", m)
