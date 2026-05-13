# Document Question&Answering with conversational Memory
A local, privacy-focused tool that lets you upload documents (PDF/TXT) and ask questions about them. The system uses semantic search to find relevant content and generates answers using Ollama with the Phi-3 model.

# Key features
- Upload PDF or TXT files
- Ask natural language questions about your document
- Conversation memory for follow-up questions
- Answers grounded ONLY in your document (no hallucinations)
- 100% local — no API keys, no data leaves your computer



# Powered by
- Ollama + Phi-3 for fast, local LLM inference
- Sentence Transformers for embeddings
- Cosine similarity for semantic search




# Why Use This?
## Problem: 
- ChatGPT doesn't know your private documents 
- Cloud APIs cost money  
- Hallucinations (AI making up answers) 

## Solution: 
- Run locally — your data never leaves your computer
- Free — Ollama + open-source models 
- Answers are grounded in your actual document 



# Installation: 
1) Install Ollama for windows: https://ollama.com 
          OR
install Ollama for Linux : curl -fsSL https://ollama.com/install.sh | sh

2) Pull The model:

ollama pull phi3


3) Install Python packages:

pip install gradio requests sentence-transformers pypdf 



# Usage
1) python app.py

Then click the public link in your browser.



# Example

Upload a file with this content:

> Machine learning is a subset of AI. There are three types: supervised, unsupervised, and reinforcement learning.

Ask: "What are the types of machine learning?"

Answer: "Supervised, unsupervised, and reinforcement learning"




# How It Works

1. Upload: a PDF or TXT file
2. Chunking: Document split into overlapping pieces (~500 words each)
3. Embeddings: Each chunk converted to a vector using Sentence Transformers
4. Storage: Chunks + embeddings stored in memory (no external DB)
5. Ask: Your question is embedded and compared to all chunks
6. Retrieve: Top 3 most semantically similar chunks are selected
7. Generate: Ollama (Phi-3) generates an answer using ONLY those chunks as context
8. Memory: Conversation history is maintained for follow-up questions


<img width="1408" height="768" alt="Project Workflow" src="https://github.com/user-attachments/assets/f05a78a1-fc2c-40d4-b97a-4f316b62c403" />


# Files
- qa_system.py: Core question answering logic 
- app.py: Gradio web interface


# Future Improvements

- Vector database (ChromaDB) for larger documents
- Hybrid search (BM25 + embeddings) for better retrieval
- Source citation (show which chunk the answer came from)
- Multi-document support



# License

MIT — Free for personal and commercial use
