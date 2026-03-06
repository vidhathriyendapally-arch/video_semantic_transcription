

import cv2
import os

# Get project root
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))

video_path = os.path.join(project_root, "data", "fightclub.mp4")
frames_folder = os.path.join(project_root, "data", "frames")

os.makedirs(frames_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_rate = 1  # extract 1 frame per second
count = 0
saved = 0

fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % int(fps * frame_rate) == 0:
        frame_path = os.path.join(frames_folder, f"frame_{saved}.jpg")
        cv2.imwrite(frame_path, frame)
        print(f"Saved {frame_path}")
        saved += 1

    count += 1

cap.release()

print("Frame extraction completed!")