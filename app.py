import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import fitz  # PyMuPDF
import docx
import io
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import cohere
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Financial Document RAG QA System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Sentence transformer model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Globals
index = None
stored_chunks = []

# Pydantic model
class QuestionRequest(BaseModel):
    question: str

# Parse uploaded document
def parse_document(uploaded_file: UploadFile) -> str:
    file_type = uploaded_file.filename.split(".")[-1].lower()
    content = uploaded_file.file.read()

    if file_type == "pdf":
        doc = fitz.open(stream=content, filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        return text

    elif file_type == "docx":
        doc = docx.Document(io.BytesIO(content))
        return "\n".join([p.text for p in doc.paragraphs])

    elif file_type == "txt":
        return content.decode("utf-8")

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

# Split text into chunks
def split_text(text, max_len=500):
    sentences = text.split('\n')
    chunks = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) < max_len:
            current_chunk += sent + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sent + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Build FAISS index
def build_faiss_index(chunks):
    global index, stored_chunks
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    stored_chunks = chunks

# Retrieve similar chunks
def retrieve_similar_chunks(question, k=5):
    if index is None or len(stored_chunks) == 0:
        return []
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    return [stored_chunks[i] for i in I[0] if i < len(stored_chunks)]

# Answer the question
def answer_question(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a financial analyst. Based on the following context, answer the question concisely and clearly:

Context:
{context}

Question:
{question}

Answer:
"""
    response = co.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=300,
        temperature=0.3,
    )
    return response.generations[0].text.strip()

# Summarize the text
def summarize_text(text):
    prompt = f"""
You are a financial analyst. Write a detailed, well-structured summary of the following document:

{text[:3000]}

Summary:
"""
    response = co.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=400,
        temperature=0.5,
    )
    return response.generations[0].text.strip()

# --- Upload endpoint ---
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        raw_text = parse_document(file)
        chunks = split_text(raw_text)
        build_faiss_index(chunks)
        return {"message": f"Document processed. {len(chunks)} chunks created."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Ask endpoint ---
@app.post("/ask")
async def ask_question(data: QuestionRequest):
    if index is None:
        raise HTTPException(status_code=400, detail="No document indexed yet. Upload a document first.")
    
    relevant_chunks = retrieve_similar_chunks(data.question)
    
    if not relevant_chunks:
        return {
            "question": data.question,
            "answer": "No relevant information found in the document.",
            "relevant_contexts": []
        }

    answer = answer_question(data.question, relevant_chunks)

    return {
        "question": data.question,
        "answer": answer,
        "relevant_contexts": relevant_chunks
    }

# --- Summarize endpoint ---
@app.get("/summarize")
async def summarize():
    if index is None or len(stored_chunks) == 0:
        raise HTTPException(status_code=400, detail="No document indexed yet. Upload a document first.")
    full_text = "\n".join(stored_chunks)
    summary = summarize_text(full_text)
    return {
        "summary": summary,
        "chunk_count": len(stored_chunks)
    }

# Run with: uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
