# Document Chat RAG System

A Retrieval-Augmented Generation (RAG) system built with **LangChain**, **OpenAI embeddings**, and **FAISS vector database** that allows users to ask questions over documents and receive grounded answers with citations.

---

## Overview

This project implements a complete **RAG pipeline**:

1. Load documents (PDF / text)
2. Split them into chunks
3. Generate embeddings
4. Store vectors in FAISS
5. Retrieve relevant chunks
6. Generate answers using an LLM

The system ensures answers are **grounded in the source documents** and includes **source citations**.

---

## Architecture

```
Documents
   ↓
Document Loaders
   ↓
Text Splitting
   ↓
Embeddings (OpenAI)
   ↓
Vector Database (FAISS)
   ↓
Retriever (MMR + MultiQuery + Compression)
   ↓
LLM (GPT)
   ↓
Answer with Citations
```

---

## Project Structure

```
document-chat-rag
│
├── src
│   ├── ingest.py       # Load documents, chunk text, build vector index
│   ├── retriever.py    # Retrieval pipeline (MMR, MultiQuery, compression)
│   └── chat.py         # RAG chain + CLI interface
│
├── data                # Input documents
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/document-chat-rag.git
cd document-chat-rag
```

Create virtual environment:

```bash
python -m venv venv
```

Activate it:

Windows

```bash
venv\Scripts\activate
```

Mac/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

---

## Adding Documents

Place your documents inside the `data/` folder.

Supported formats:

- `.pdf`
- `.txt`

---

## Build Vector Index

Run ingestion:

```bash
python src/ingest.py
```

This will:

- Load documents
- Split them into chunks
- Generate embeddings
- Store vectors in FAISS

---

## Start the Chat System

```bash
python src/chat.py
```

You can now ask questions about your documents.

Example:

```
Ask: What are the key points discussed in the document?
```

The system retrieves relevant chunks and generates an answer with citations.

---

## Features

- LangChain-based RAG pipeline
- OpenAI embeddings
- FAISS vector database
- Advanced retrieval techniques:
  - MMR (Maximum Marginal Relevance)
  - MultiQueryRetriever
  - Contextual compression

- Source citations in answers
- CLI-based document chat

---

## Technologies Used

- Python
- LangChain
- OpenAI API
- FAISS
- dotenv

---

## Future Improvements

- Switch to production vector database (Qdrant)
- Add reranking models
- Add evaluation framework for RAG quality
- Add web interface (Streamlit / FastAPI)
- Implement agentic RAG workflows

---

## Author

Sri Krishna A
Associate Data Scientist at Shell

This project is part of my work exploring **LLM systems, RAG architectures, and AI engineering workflows**.
