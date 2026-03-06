import os

# Get project root directory
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# File paths
audio_path = os.path.join(project_root, "data", "audio_transcription.txt")
video_path = os.path.join(project_root, "data", "video_transcription.txt")
output_path = os.path.join(project_root, "data", "semantic_transcription.txt")

print("Reading audio transcription...")
print("Reading video transcription...")

# Read files
with open(audio_path, "r", encoding="utf-8") as f:
    audio_lines = f.readlines()

with open(video_path, "r", encoding="utf-8") as f:
    video_lines = f.readlines()

combined_lines = []

max_len = max(len(audio_lines), len(video_lines))

for i in range(max_len):

    audio_text = audio_lines[i].strip() if i < len(audio_lines) else ""
    video_text = video_lines[i].strip() if i < len(video_lines) else ""

    line = f"Audio: {audio_text} | Video: {video_text}"

    combined_lines.append(line)
    print(line)

# Save combined file
with open(output_path, "w", encoding="utf-8") as f:
    for line in combined_lines:
        f.write(line + "\n")

print("\nSemantic transcription saved at:")
print(output_path)