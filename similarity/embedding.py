import os
from sentence_transformers import SentenceTransformer
import numpy as np

# project root
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# text file path
text_path = os.path.join(project_root, "data", "clean_text.txt")

# read text
with open(text_path, "r", encoding="utf-8") as f:
    text = f.read()

# load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# create embedding
embedding = model.encode(text)

print("✅ Embedding vector created!")

# show only first 10 numbers
print("First 10 values:", embedding[:10])

# save embedding
output_path = os.path.join(project_root, "data", "embedding.npy")
np.save(output_path, embedding)

print("✅ Embedding saved at:", output_path)