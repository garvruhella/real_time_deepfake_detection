import os, argparse, time, cv2, numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

def extract_for_video(video_path, out_folder, mtcnn, frame_step=5, max_frames=None):
    os.makedirs(out_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0; saved = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % frame_step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb)
            if boxes is not None and len(boxes) > 0:
                x1,y1,x2,y2 = [int(v) for v in boxes[0]]
                face = rgb[y1:y2, x1:x2]
                if face.size != 0:
                    Image.fromarray(face).save(os.path.join(out_folder, f"{saved:06d}.png"))
                    saved += 1
                    if max_frames and saved >= max_frames: break
        idx += 1
    cap.release()
    return saved

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="datasets/faces_by_video")
    parser.add_argument("--frame_step", type=int, default=5)
    parser.add_argument("--max_per_video", type=int, default=500)
    args = parser.parse_args()

    device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=True, device=device)
    vids = [v for v in os.listdir(args.input_dir) if v.lower().endswith(('.mp4','.mov','.avi'))]

    for v in vids:
        name = os.path.splitext(v)[0]
        out = os.path.join(args.output_dir, name)
        print(f"🎥 Extracting faces from {v}")
        n = extract_for_video(os.path.join(args.input_dir, v), out, mtcnn, args.frame_step, args.max_per_video)
        print(f"✅ {n} faces saved to {out}")

if __name__ == "__main__":
    main()
