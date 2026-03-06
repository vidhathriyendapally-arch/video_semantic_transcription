import os
import sys
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

print("Loading BLIP model for video captioning...")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# -----------------------------
# PROJECT ROOT
# -----------------------------
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# -----------------------------
# INPUT VIDEO
# -----------------------------
if len(sys.argv) < 2:
    print("Usage: python video_action_transcription.py <video_file.mp4>")
    sys.exit()

video_file = sys.argv[1]

video_path = os.path.join(project_root, "data", video_file)

# -----------------------------
# OUTPUT FILE
# -----------------------------
output_path = os.path.join(project_root, "data", "video_transcription.txt")

# -----------------------------
# OPEN VIDEO
# -----------------------------
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video FPS:", fps)

# frame interval for 2 seconds
frame_interval = fps * 2

frame_count = 0
timestamp = 0

output_lines = []

print("Starting 2-second video captioning...\n")

while True:

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    ret, frame = cap.read()

    if not ret:
        break

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)

    line = f"[{timestamp:02d} sec] {caption}"

    print(line)

    output_lines.append(line)

    frame_count += frame_interval
    timestamp += 2

cap.release()

# -----------------------------
# SAVE OUTPUT
# -----------------------------
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("\nVideo transcription saved at:", output_path)