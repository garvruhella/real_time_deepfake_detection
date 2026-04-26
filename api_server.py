import io
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms, models
from PIL import Image

app = FastAPI()

# === DEVICE ===
DEVICE = torch.device("cpu")  # keep CPU for stability

# === LOAD MODEL ===
model = models.resnet18(weights=None)

# MUST match training architecture
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 1)
)

model.load_state_dict(torch.load("best_classifier.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully")

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === API ROUTE ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()

        label = "REAL" if prob > 0.5 else "FAKE"

        return {
            "prediction": label,
            "confidence": round(prob if prob > 0.5 else 1 - prob, 4)
        }

    except Exception as e:
        return {"error": str(e)}