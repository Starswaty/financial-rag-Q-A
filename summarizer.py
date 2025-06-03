import cohere
import os
from dotenv import load_dotenv

load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

def summarize_text(text):
    prompt = f"""
Summarize the following financial document clearly and concisely:

{text[:3000]}

Summary:
"""
    response = co.generate(
        model="command-small",
        inputs=[prompt],
        max_tokens=256,
        temperature=0.5,
        input_type="text"
    )
    return response.generations[0].text.strip()
