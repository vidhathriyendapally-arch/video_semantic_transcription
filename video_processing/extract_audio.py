import os
import sys
import whisper
from moviepy.editor import VideoFileClip

print("Loading Whisper model...")

# Load whisper model
model = whisper.load_model("base")

# Get project root
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# ----------------------------
# INPUT VIDEO FILE
# ----------------------------

if len(sys.argv) < 2:
    print("Usage: python transcribe_audio.py <video_file.mp4>")
    sys.exit()

video_file = sys.argv[1]

video_path = os.path.join(project_root, "data", video_file)

# ----------------------------
# OUTPUT PATHS
# ----------------------------

audio_path = os.path.join(project_root, "data", "audio.wav")
text_path = os.path.join(project_root, "data", "audio_transcription.txt")

# ----------------------------
# STEP 1 : EXTRACT AUDIO
# ----------------------------

print("\nExtracting audio from video...")

video = VideoFileClip(video_path)
video.audio.write_audiofile(audio_path)

print("Audio extracted at:", audio_path)

# ----------------------------
# STEP 2 : TRANSCRIBE AUDIO
# ----------------------------

print("\nTranscribing audio using Whisper...")

result = model.transcribe(audio_path)

segments = result["segments"]

transcription_lines = []
previous_end = 0

for seg in segments:

    start = seg["start"]
    end = seg["end"]
    text = seg["text"].strip()

    pause = start - previous_end

    if pause > 1.5:
        pause_line = f"(pause_time: {round(pause,2)} sec) (no audio detected)"
        transcription_lines.append(pause_line)
        print(pause_line)

    line = f"[{round(start,2)}s - {round(end,2)}s] {text}"

    transcription_lines.append(line)
    print(line)

    previous_end = end

# ----------------------------
# STEP 3 : SAVE TRANSCRIPTION
# ----------------------------

with open(text_path, "w", encoding="utf-8") as f:
    for line in transcription_lines:
        f.write(line + "\n")

print("\nAudio transcription saved at:")
print(text_path)