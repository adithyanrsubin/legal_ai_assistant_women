from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="womens_legal_docs")

question = "What should I do if my colleague is harassing me at work?"
question_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3,
)

print(f"Question: {question}\n")
for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"--- Result {i+1} (source: {meta['source']}) ---")
    print(doc[:300])
    print()