import re
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


# 1. Clean text properly
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)   # remove extra spaces/newlines
    return text.strip()


# 2. Load PDF
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    return clean_text(text)


# 3. Split into chunks (sentence-based better)
def split_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# 4. Create FAISS index
def create_vector_store(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, model


# 5. Answer query (clean output)
def answer_query(index, embed_model, chunks, query):
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)

    best_chunk = chunks[I[0][0]]

    # Try to extract most relevant sentence
    sentences = best_chunk.split(".")
    for sentence in sentences:
        if any(word.lower() in sentence.lower() for word in query.split()):
            return sentence.strip()

    return best_chunk.strip()


# 6. Main
def main():
    print("Loading PDF...")
    text = load_pdf("sample.pdf")

    print("Splitting text...")
    chunks = split_text(text)

    print("Creating vector database...")
    index, embed_model = create_vector_store(chunks)

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        answer = answer_query(index, embed_model, chunks, query)
        print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
