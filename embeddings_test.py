from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence = "A woman facing harassment at her workplace can file a complaint."
embedding = model.encode(sentence)

print(f"Embedding length: {len(embedding)}")
print(f"First 10 numbers: {embedding[:10]}")