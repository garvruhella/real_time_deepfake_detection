# check_dataset_stats.py
import os
root = "datasets/training_data"
for split in ['train','val','test']:
    print(f"=== {split} ===")
    for cls in ['real','fake']:
        p = os.path.join(root, split, cls)
        n = 0
        if os.path.isdir(p):
            n = len([f for f in os.listdir(p) if f.lower().endswith(('.jpg','.png'))])
        print(f"{cls}: {n}")
