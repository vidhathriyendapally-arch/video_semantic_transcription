import cv2
import json
from deepface import DeepFace
import textwrap
import os

# -----------------------------
# FILE PATHS
# -----------------------------
video_path = r"C:\Users\HP\OneDrive\Desktop\video_similarity_project\data\fightclub.mp4"
transcription_path = r"C:\Users\HP\OneDrive\Desktop\video_similarity_project\data\transcription.json"
semantic_path = r"C:\Users\HP\OneDrive\Desktop\video_similarity_project\data\semantic_transcription.json"  # optional
output_path = r"C:\Users\HP\OneDrive\Desktop\video_similarity_project\data\output_video_netflix.avi"

# -----------------------------
# CHECK FILES EXIST
# -----------------------------
for path in [video_path, transcription_path]:
    if not os.path.exists(path):
        print(f"Error: {path} not found!")
        exit()

# -----------------------------
# LOAD TRANSCRIPTIONS
# -----------------------------
with open(transcription_path, "r") as f:
    transcription = json.load(f)

semantic_transcription = {}
if os.path.exists(semantic_path):
    with open(semantic_path, "r") as f:
        semantic_transcription = json.load(f)

# -----------------------------
# OPEN VIDEO
# -----------------------------
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0

# -----------------------------
# PROCESS VIDEO
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    time_sec = int((frame_count / fps) // 2) * 2
   

    # -----------------------------
    # GET CURRENT SUBTITLE
    # -----------------------------
    subtitle_text = ""
    semantic_text = ""
    segment_id = None
    for segment in transcription:
        if segment["start"] <= time_sec <= segment["end"]:
            subtitle_text = segment["text"]
            segment_id = segment.get("id", None)
            if segment_id and segment_id in semantic_transcription:
                semantic_text = semantic_transcription[segment_id]
            break

    # -----------------------------
    # EMOTION DETECTION
    # -----------------------------
    emotion = "Detecting..."
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            emotion = result[0]["dominant_emotion"]
        else:
            emotion = result["dominant_emotion"]
    except:
        emotion = "No Face"

    # -----------------------------
    # PREPARE SUBTITLE LINES
    # -----------------------------
    lines = []
    if emotion:
        lines.append(f"[{emotion}] {subtitle_text}")   # emotion + original
    elif subtitle_text:
        lines.append(subtitle_text)
    if semantic_text:
        lines.append(f"({semantic_text})")            # semantic below

    # Wrap lines to fit video width
    wrapped_lines = []
    for line in lines:
        wrapped_lines.extend(textwrap.wrap(line, width=50))  # adjust width as needed

    # -----------------------------
    # DRAW SEMI-TRANSPARENT SUBTITLE BOX
    # -----------------------------
    y0 = height - 120
    dy = 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, y0 - 30), (width-20, height-20), (0, 0, 0), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # -----------------------------
    # DRAW TEXT
    # -----------------------------
    for i, line in enumerate(wrapped_lines):
        y = y0 + i*dy
        cv2.putText(frame, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # -----------------------------
    # WRITE FRAME
    # -----------------------------
    out.write(frame)
    frame_count += 1

# -----------------------------
# RELEASE RESOURCES
# -----------------------------
cap.release()
out.release()

print("✅ Netflix-style subtitles video created!")
print("Saved at:", output_path)