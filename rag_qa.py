import cohere
import os
from dotenv import load_dotenv

load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

def answer_question(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a financial analyst. Based on the following context, answer the question:

Context:
{context}

Question:
{question}

Answer:
"""
    response = co.generate(
        model="command-small",        # Free-tier model
        inputs=[prompt],              # inputs must be a list of strings
        max_tokens=300,
        temperature=0.3
    )
    return response.generations[0].text.strip()
