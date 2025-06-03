import faiss
import numpy as np

index = None
stored_chunks = []

def store_embeddings(embeddings, chunks):
    global index, stored_chunks
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    stored_chunks = chunks

def retrieve_similar_chunks(query, k=5):
    from embedder import embed_chunks
    if index is None:
        return ["No documents indexed"]
    query_vec = embed_chunks([query])
    distances, indices = index.search(query_vec.astype('float32'), k)
    return [stored_chunks[i] for i in indices[0] if i < len(stored_chunks)]