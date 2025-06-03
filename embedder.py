import cohere
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

def embed_chunks(chunks):
    response = co.embed(texts=chunks, model="embed-english-v3.0")
    return np.array(response.embeddings)
