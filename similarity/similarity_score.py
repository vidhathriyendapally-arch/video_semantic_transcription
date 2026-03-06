from sentence_transformers import SentenceTransformer, util

text1 = open("data/clean_text.txt").read()
text2 = open("data/clean_text2.txt").read()   # transcription of second video

model = SentenceTransformer("all-MiniLM-L6-v2")

emb1 = model.encode(text1)
emb2 = model.encode(text2)

score = util.cos_sim(emb1, emb2)

print("Similarity Score:", score.item())