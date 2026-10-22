from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import re

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def chunk_by_sections(text, min_chunk_size=150, max_chunk_size=1500):
    parts = re.split(r'\n(?=\d{1,3}[A-Z]?\.\s[A-Z])', text)

    chunks = []
    for part in parts:
        part = part.strip()
        if len(part) < min_chunk_size:
            continue
        if len(part) > max_chunk_size:
            start = 0
            while start < len(part):
                end = start + max_chunk_size
                sub_chunk = part[start:end].strip()
                if len(sub_chunk) >= min_chunk_size:
                    chunks.append(sub_chunk)
                start += max_chunk_size - 200
        else:
            chunks.append(part)
    return chunks

documents = [
    ("../data/legal_documents/workplace_harassment.pdf", "POSH Act 2013"),
    ("../data/legal_documents/domesticviolence.pdf", "Domestic Violence Act 2005"),
    ("../data/legal_documents/dowry_prohibition_act_1961.pdf", "Dowry Prohibition Act 1961"),
    ("../data/legal_documents/immoral_traffic_prevention_act_1956.pdf", "Immoral Traffic Prevention Act 1956"),
    ("../data/legal_documents/medical_termination_pregnancy_act_1971.pdf", "Medical Termination of Pregnancy Act 1971"),
    ("../data/legal_documents/sati_prevention_act_1987.pdf", "Commission of Sati Prevention Act 1987"),
    ("../data/legal_documents/indecent_representation_women_act_1986.pdf", "Indecent Representation of Women Act 1986"),
]

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Setting up ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")

try:
    client.delete_collection(name="womens_legal_docs")
    print("Cleared old collection (rebuilding with improved chunking).")
except Exception:
    pass

collection = client.get_or_create_collection(name="womens_legal_docs")

for file_path, doc_name in documents:
    print(f"Processing: {doc_name}")
    text = extract_text_from_pdf(file_path)
    chunks = chunk_by_sections(text)

    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_name} for _ in chunks]
    embeddings = model.encode(chunks).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    print(f"  Added {len(chunks)} chunks from {doc_name}")

print(f"\nTotal chunks in database: {collection.count()}")