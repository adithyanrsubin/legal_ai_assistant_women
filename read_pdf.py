from pypdf import PdfReader

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

posh_text = extract_text_from_pdf("../data/legal_documents/workplace_harassment.pdf")
posh_chunks = chunk_text(posh_text)

print(f"POSH Act text length: {len(posh_text)} characters")
print(f"Number of chunks: {len(posh_chunks)}")
print("---- First chunk ----")
print(posh_chunks[0])
print("---- Second chunk (should overlap with the end of the first) ----")
print(posh_chunks[1])