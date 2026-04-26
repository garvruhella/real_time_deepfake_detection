import os
from glob import glob

def get_video_id(path):
    # expects .../class/<video_id>/<frame>.jpg
    return path.split(os.sep)[-2]

train = glob("datasets/final_data/train/*/*/*.jpg")
val   = glob("datasets/final_data/val/*/*/*.jpg")

train_vids = set(get_video_id(p) for p in train)
val_vids   = set(get_video_id(p) for p in val)

overlap = train_vids.intersection(val_vids)

print(f"Train videos: {len(train_vids)}")
print(f"Val videos:   {len(val_vids)}")
print(f"Overlapping videos: {len(overlap)}")