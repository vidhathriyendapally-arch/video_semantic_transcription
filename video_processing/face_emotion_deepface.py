import cv2
from deepface import DeepFace
import os
import json

print("Script started")

# project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

video_path = os.path.join(project_root, "data", "fightclub.mp4")

print("Video path:", video_path)

cap = cv2.VideoCapture(video_path)

frame_count = 0
emotion_counts = {}

print("Processing video...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # analyze every 20th frame
    if frame_count % 20 == 0:

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            emotion = result[0]["dominant_emotion"]

            print(f"Frame {frame_count} → {emotion}")

            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        except:
            pass

cap.release()

if emotion_counts:
    final_emotion = max(emotion_counts, key=emotion_counts.get)
else:
    final_emotion = "no emotion detected"

print("\nFinal Emotion:", final_emotion)

result = {
    "video": "fightclub.mp4",
    "dominant_emotion": final_emotion,
    "emotion_counts": emotion_counts
}

output_path = os.path.join(project_root, "data", "face_emotion.json")

with open(output_path, "w") as f:
    json.dump(result, f, indent=4)

print("Saved to:", output_path)
