RAG Chatbot
A Retrieval-Augmented Generation chatbot that uses sentence-aware chunked PDF content, FAISS vector search, and Groq-hosted LLMs (llama-3.1-8b-instant) to provide accurate, source-backed answers via a Streamlit interface.


1. Project Architecture and Flow:
    a. PDF processing
        - PDF is stored in /data/
        - processpdf.py reads and splits into 100–300 word chunks (sentence-aware) and saves the chunks as chunks.json in /chunks/
    b. Embedding + indexing
        - genEmbeds.py uses embedding model (all-MiniLM-L6-v2) to generate embeddings using chunks
        - stores them in FAISS index as /vectordb/faiss_index.index
    c. Retrieval
        - retriever.py takes user query, converts it into embeds and retrieves top k (default k = 5) relevant chunks
    d. Generation
        - generator.py builds a prompt using retrieved chunks and the query
        - sends it to the llm via API
        - streams response back token by token
    e. UI
        - app.py is the streamlit app for UI
        - Users input the queries and view streamed answers and their source chunks.


2. Steps to run
    a.Clone repo and install dependencies
    b. Create .env and add your API key
    c. If you want to use a new PDF:
        - add your pdf in data folder
        - modify the name of pdf in processpdf.py
        - run processpdf.py
        - run genEmbeds.py
    d. Run app.py using streamlit

3. Models choices
    a. Model used to generate embeddings : all-MiniLM-L6-v2
    b. LLM model used to answer the query : llama-3.1-8b-instant via Groq API
    Reason:
     - easy to set up on windows
     - No need to download model and set it up (using Ollama)
     - fast repsonse


4. Sample queries

a.
![alt text](<Screenshot 2025-07-05 153149.png>)

b.
![alt text](<Screenshot 2025-07-05 153205.png>)

c.
![alt text](<Screenshot 2025-07-05 153205-1.png>)
