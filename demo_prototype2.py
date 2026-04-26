import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

VIDEO_PATH = "videos_to_use/test_video4.mp4"
OUTPUT_PATH = "demo_output.mp4"
MODEL_PATH = "best_classifier.pth"

DEVICE = torch.device("cpu")

# model
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 1)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()

    label = "REAL" if prob > 0.5 else "FAKE"
    color = (0,255,0) if label=="REAL" else (0,0,255)

    cv2.putText(frame, f"{label} {prob:.2f}", (30,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    out.write(frame)

cap.release()
out.release()

print("Done. Output saved as demo_output.mp4")