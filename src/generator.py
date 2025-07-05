from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = OpenAI(
    api_key = groq_api_key,
    base_url= "https://api.groq.com/openai/v1",

)

# Builds prompt using the context_chunks and the user query, this function will be used in app.py
def build_prompt(context_chunks, query):
    context = "\n\n".join(context_chunks)
    return (
        f"You are a helpful assistant. use only the context (no external information) to answer the query and if the context doesnt contain the answer, let the user know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    ),context


model_name = "llama-3.1-8b-instant"



def stream_llm(prompt):
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=512,
        stream=True,  # Enable streaming
    )
    # so that i can show streaming responses
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
