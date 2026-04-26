import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

# --- Config ---
video_path = "032_944.mp4"  # change path if needed
output_path = "results_detector/output_detected.mp4"
model_path = "results_detector/best_detector.pth"

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running inference on {device}...")

# --- Load face detector (MTCNN) ---
mtcnn = MTCNN(keep_all=True, device=device)

# --- Define transform (convert to PIL then Tensor) ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- Load trained detector ---
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# --- Video I/O setup ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs(os.path.dirname(output_path), exist_ok=True)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# --- Process frames ---
print(f"Processing {frame_count} frames...")
for _ in tqdm(range(frame_count), ncols=100):
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame for MTCNN
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            face = rgb_frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # --- Convert NumPy -> PIL ---
            face_pil = Image.fromarray(face)

            # --- Transform & predict ---
            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                output = torch.sigmoid(model(face_tensor)).item()

            # --- Label & Draw ---
            label = "FAKE" if output < 0.5 else "REAL"
            color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
            prob = output if label == "REAL" else 1 - output

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({prob*100:.1f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)

cap.release()
out.release()
print(f"\n✅ Inference complete! Output saved to: {output_path}")
