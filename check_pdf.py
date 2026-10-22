from pypdf import PdfReader

reader = PdfReader("../data/legal_documents/criminal_law_amendment_act_2013.pdf")
print(f"Number of pages: {len(reader.pages)}")

text = ""
for page in reader.pages:
    text += page.extract_text()

print(f"Total characters extracted: {len(text)}")
print("---- First 500 characters ----")
print(text[:500])
