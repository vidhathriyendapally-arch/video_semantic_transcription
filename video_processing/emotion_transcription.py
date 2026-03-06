import os
import whisper
import torch
import librosa
import numpy as np
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, HubertForSequenceClassification

# 1. Load Whisper model
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

# 2. Load emotion detection model from HuggingFace (auto download)
print("Loading emotion detection model...")
processor = Wav2Vec2Processor.from_pretrained("superb/hubert-large-superb-er")
model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")

# 3. Define project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
audio_path = os.path.join(project_root, "data", "audio.wav")
output_path = os.path.join(project_root, "data", "audio_emotion_transcription.txt")

# 4. Load audio
print("Loading audio...")
audio = AudioSegment.from_wav(audio_path)

# Split audio into 5-second chunks
chunk_length_ms = 5000
chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

final_transcription = ""

print(f"Total chunks: {len(chunks)}")

# 5. Process each chunk
for idx, chunk in enumerate(chunks):

    print(f"Processing chunk {idx+1}/{len(chunks)}")

    chunk_path = os.path.join(project_root, f"temp_chunk_{idx}.wav")
    chunk.export(chunk_path, format="wav")

    # Whisper transcription
    result = whisper_model.transcribe(chunk_path)
    text = result["text"].strip()

    # Emotion detection
    speech, rate = librosa.load(chunk_path, sr=16000)

    inputs = processor(
        speech,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_id = torch.argmax(logits, dim=-1).item()
    emotion_label = model.config.id2label[predicted_id]

    print("Text:", text)
    print("Emotion:", emotion_label)

    final_transcription += f"{text} [{emotion_label}]\n"

    # delete temp chunk
    os.remove(chunk_path)

# 6. Save transcription
with open(output_path, "w", encoding="utf-8") as f:
    f.write(final_transcription)

print("\n✅ Emotion transcription saved to:")
print(output_path)