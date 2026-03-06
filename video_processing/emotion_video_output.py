import cv2
import json
from deepface import DeepFace

# -----------------------------
# FILE PATHS
# -----------------------------
video_path = "data/input_video.mp4"
transcription_path = "data/transcription.json"
output_path = "data/output_video.mp4"

# -----------------------------
# LOAD TRANSCRIPTION
# -----------------------------
with open(transcription_path, "r") as f:
    transcription = json.load(f)

# -----------------------------
# OPEN VIDEO
# -----------------------------
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_path = "data/output_video.avi"  # change extension to .avi for better compatibility
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0

# -----------------------------
# PROCESS VIDEO
# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    time_sec = frame_count / fps

    subtitle_text = ""

    # -----------------------------
    # GET SUBTITLE BASED ON TIME
    # -----------------------------
    for segment in transcription:

        if segment["start"] <= time_sec <= segment["end"]:
            subtitle_text = segment["text"]
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
    # DRAW EMOTION TEXT
    # -----------------------------
    cv2.putText(
        frame,
        f"Emotion: {emotion}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # -----------------------------
    # DRAW SUBTITLE
    # -----------------------------
    cv2.putText(
        frame,
        subtitle_text,
        (50, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # -----------------------------
    # SAVE FRAME
    # -----------------------------
    out.write(frame)

    frame_count += 1

# -----------------------------
# RELEASE
# -----------------------------
cap.release()
out.release()

print("✅ Video created successfully!")
print("Saved at:", output_path)