import cv2
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import json
import os

# Load pretrained model
model = timm.create_model("resnet18", pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
video_path = os.path.join(project_root, "data", "fightclub.mp4")
output_file = os.path.join(project_root, "data", "face_emotion.json")

cap = cv2.VideoCapture(video_path)

emotion_counts = {}

print("Processing video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)

    emotion = torch.argmax(outputs).item()
    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

cap.release()

if emotion_counts:
    final_emotion = max(emotion_counts, key=emotion_counts.get)
else:
    final_emotion = "No frames processed"

with open(output_file, "w") as f:
    json.dump({"face_emotion": str(final_emotion)}, f, indent=4)

print("✅ Face Emotion:", final_emotion)