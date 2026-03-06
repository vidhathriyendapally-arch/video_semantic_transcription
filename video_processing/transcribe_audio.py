import os
import sys
import whisper
import subprocess

print("Loading Whisper model...")
model = whisper.load_model("base")

# -----------------------------
# PROJECT ROOT
# -----------------------------
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# -----------------------------
# INPUT VIDEO FILE
# -----------------------------
if len(sys.argv) < 2:
    print("Usage: python transcribe_audio.py <video_file.mp4>")
    sys.exit()

video_file = sys.argv[1]
video_path = os.path.join(project_root, "data", video_file)

# -----------------------------
# AUDIO OUTPUT
# -----------------------------
audio_path = os.path.join(project_root, "data", "audio.wav")

# -----------------------------
# TRANSCRIPTION OUTPUT
# -----------------------------
output_path = os.path.join(project_root, "data", "audio_transcription.txt")

# -----------------------------
# EXTRACT AUDIO FROM VIDEO
# -----------------------------
print("\nExtracting audio from video...")

command = [
    "ffmpeg",
    "-y",
    "-i", video_path,
    "-ar", "16000",
    "-ac", "1",
    audio_path
]

subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Audio extracted at:", audio_path)

# -----------------------------
# TRANSCRIBE AUDIO
# -----------------------------
print("\nTranscribing audio...")

result = model.transcribe(audio_path)

segments = result["segments"]

transcription_lines = []

previous_end = 0

# -----------------------------
# PROCESS AUDIO SEGMENTS
# -----------------------------
for seg in segments:

    start = seg["start"]
    end = seg["end"]
    text = seg["text"].strip()

    # Calculate pause duration
    pause = start - previous_end

    if pause > 1.5:
        pause_seconds = int(round(pause))
        pause_line = f"(pause time: {pause_seconds:02d} s) (audio not detected)"
        transcription_lines.append(pause_line)
        print(pause_line)

    # Speech line
    speech_line = f"[{round(start,2)}s - {round(end,2)}s] {text}"
    transcription_lines.append(speech_line)
    print(speech_line)

    previous_end = end

# -----------------------------
# SAVE TRANSCRIPTION
# -----------------------------
with open(output_path, "w", encoding="utf-8") as f:
    for line in transcription_lines:
        f.write(line + "\n")

print("\nAudio transcription saved at:")
print(output_path)
