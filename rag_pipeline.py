from sentence_transformers import SentenceTransformer
import chromadb
import ollama

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="womens_legal_docs")

def get_answer(question, n_results=4):
    question_embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[question_embedding], n_results=n_results)
    retrieved_chunks = results["documents"][0]
    sources = results["metadatas"][0]

    context = "\n\n".join(
        f"[Source: {meta['source']}]\n{chunk}"
        for chunk, meta in zip(retrieved_chunks, sources)
    )

    prompt = f"""You are a legal assistant that helps women understand their legal rights in India, based strictly on the legal text provided below.

Rules you must follow:
1. Only refuse if the question is CLEARLY about something unrelated to women's legal rights in India (for example, explicitly about men's rights, general chit-chat, or a completely different topic like cooking or sports). If a question is short, vague, or just names a topic (like "immoral traffic" or "dowry"), treat it as the person wanting to understand their rights on that topic, and answer using the legal text provided instead of refusing.
2. Interpret the question the way a real person would describe their situation, even if they don't use legal terms. For example, "someone keeps making rude comments about my body at college" should be understood as sexual harassment in an educational institution, and "my husband hits me" should be understood as domestic violence, even though those exact words aren't used.
3. If the question is clearly out of scope per Rule 1, respond with exactly: "I'm designed to help specifically with women's legal rights in India. This question falls outside that scope."
4. If the question IS about women's rights but the provided legal text doesn't contain the answer, say: "I don't have enough information in my current documents to answer that accurately."
5. Otherwise, answer using ONLY the legal text provided. Do not use outside legal knowledge, but you MAY use everyday reasoning to connect the person's situation to the correct legal category.
6. Keep your answer SHORT and focused - 3 to 5 sentences maximum, unless the question genuinely requires listing multiple specific points, in which case use a brief bullet list instead of long paragraphs.
7. Write in plain, warm, supportive language a non-lawyer can understand. Briefly acknowledge their situation before explaining their rights and options. Avoid repeating the question back at the start.

Legal text:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
    answer = response['message']['content']
    used_sources = list(set(meta['source'] for meta in sources))
    return answer, used_sources

if __name__ == "__main__":
    question = "What should I do if my colleague is harassing me at work?"
    answer, sources = get_answer(question)
    print(f"Question: {question}\n\nAnswer:\n{answer}\n\nSources used: {sources}")