import cv2
from fer import FER
import os
import json
print("Script started")
# Get project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Video path
video_path = os.path.join(project_root, "data", "fightclub.mp4")

print("Script started")
print("Video path:", video_path)

# Open video
cap = cv2.VideoCapture(video_path)
print("Video opened:", cap.isOpened())

print("Processing video...")

# FER detector
detector = FER(mtcnn=True)

frame_count = 0
emotion_counts = {}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Process every 15th frame
    if frame_count % 15 == 0:

        emotions = detector.detect_emotions(frame)

        if emotions:

            emotion_data = emotions[0]["emotions"]

            # Get highest emotion
            dominant_emotion = max(emotion_data, key=emotion_data.get)

            print(f"Frame {frame_count} → {dominant_emotion}")

            # Count emotions
            emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1

cap.release()

# Final emotion
if emotion_counts:
    final_emotion = max(emotion_counts, key=emotion_counts.get)
else:
    final_emotion = "no emotion detected"

print("\nFinal Emotion:", final_emotion)

# Save result
result = {
    "video": "fightclub.mp4",
    "dominant_emotion": final_emotion,
    "emotion_counts": emotion_counts
}

output_path = os.path.join(project_root, "data", "face_emotion.json")

with open(output_path, "w") as f:
    json.dump(result, f, indent=4)

print("Result saved to:", output_path)