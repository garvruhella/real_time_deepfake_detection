import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from facenet_pytorch import MTCNN
from tqdm import tqdm
from PIL import Image

# =========================
# CONFIG (CPU FOR STABILITY)
# =========================
DEVICE = torch.device("cpu")   # FORCE CPU (fixes CUDA crashes)

VIDEO_PATH = "videos_to_use/test_video4.mp4"
OUTPUT_PATH = "demo_output4.mp4"
MODEL_PATH = "results_detector/best_detector.pth"

print(f"🚀 Running demo on {DEVICE}...")

# =========================
# LOAD MODEL
# =========================
model = models.resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)

model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully.")

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =========================
# FACE DETECTOR (CPU)
# =========================
mtcnn = MTCNN(keep_all=False, device="cpu")

# =========================
# VIDEO LOAD
# =========================
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"❌ Video not found: {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"❌ Could not open video: {VIDEO_PATH}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

print(f"🎥 Processing {total_frames} frames...")

# =========================
# PROCESS VIDEO
# =========================
for _ in tqdm(range(total_frames), desc="Analyzing video"):
    ret, frame = cap.read()
    if not ret:
        break

    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)
    except Exception:
        boxes = None

    if boxes is not None:
        for box in boxes:
            try:
                x1, y1, x2, y2 = [int(v) for v in box]

                # Clamp to frame boundaries (IMPORTANT FIX)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                face_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = model(face_tensor)
                    prob = torch.sigmoid(output).item()

                label = "REAL" if prob > 0.5 else "FAKE"
                conf = prob if prob > 0.5 else 1 - prob
                color = (0, 255, 0) if label == "REAL" else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ({conf*100:.1f}%)",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            except Exception:
                continue

    out.write(frame)

# =========================
# CLEANUP
# =========================
cap.release()
out.release()

print(f"✅ Demo video saved as: {OUTPUT_PATH}")