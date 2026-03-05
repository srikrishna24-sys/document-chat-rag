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
```
