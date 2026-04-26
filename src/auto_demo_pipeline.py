import os
import subprocess
import time

def run_command(cmd, desc):
    print(f"\n>>> {cmd}")
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        print(f"✅ {desc} completed in {(time.time()-start)/60:.1f} min.")
    else:
        print(f"⚠️ Command failed: {cmd}")
        exit(1)

def main():
    print("🚀 Auto deepfake demo pipeline starting...\n")

    # Step 0: Ensure directories exist
    os.makedirs("videos_to_use", exist_ok=True)
    print("👉 Place 2 real and 2 fake videos inside 'videos_to_use/'")
    print("   e.g., real1.mp4, real2.mp4, fake1.mp4, fake2.mp4")

    input("✅ Press ENTER when videos are ready...")

    # Step 1: Extract faces
    if not os.path.exists("datasets/faces_by_video"):
        run_command(
            "python src/extract_faces_by_video.py --input_dir videos_to_use --output_dir datasets/faces_by_video --frame_step 5 --max_per_video 500",
            "Face extraction"
        )
    else:
        print("⏩ Faces already extracted. Skipping...")

    # Step 2: Build detector dataset
    if not os.path.exists("datasets/detector_data/train/real"):
        run_command(
            "python src/build_video_dataset.py --faces_root datasets/faces_by_video --out_root datasets/detector_data --train_ratio 0.8",
            "Dataset building"
        )
    else:
        print("⏩ Detector dataset already exists. Skipping...")

    # Step 3: Train detector
    if not os.path.exists("results_detector/best_detector.pth"):
        run_command(
            "python -m src.train_detector",
            "Detector training"
        )
    else:
        print("⏩ Detector already trained. Skipping...")

    # Step 4: Run demo
    if not os.path.exists("videos_to_use/test_video.mp4"):
        print("⚠️ Please add a test video named 'test_video.mp4' inside 'videos_to_use/'.")
        return

    run_command(
        "python -m src.demo_prototype",
        "Demo prototype (deepfake detection)"
    )

    print("\n🎬 All steps completed successfully!")
    print("➡️ Output video saved as 'demo_output.mp4' in project root.\n")


if __name__ == "__main__":
    main()
