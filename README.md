# Thunderbird AI Email Assistant — Backend

Backend implementation for an AI-powered email assistant that generates and
refines email responses using retrieval-augmented generation (RAG).

This project was developed as part of a professional software project.
This repository contains the **backend implementation only**; the
Thunderbird extension/frontend code is not included.

---

## Overview

The backend provides APIs for processing email requests, ingesting reference
emails and user-provided datasets, generating embeddings, retrieving relevant
context from a vector database, and generating/refining email responses with
an LLM.

The system uses a session-based architecture so that reference emails and
uploaded datasets can be isolated per user/session.

### Core capabilities

- RAG-based email response generation
- Reference email retrieval for writing-style adaptation
- Dataset ingestion from PDF and CSV files
- Text extraction and chunking
- Vector embedding generation
- Pinecone vector storage and similarity search
- Session-scoped vector namespaces
- LLM-based response generation
- LLM-based response refinement
- Real-time embedding/upload progress through WebSockets
- REST API backend for communication with the Thunderbird client

---

## Architecture

The backend is built around two retrieval sources:

### 1. Reference Emails

Reference emails represent the user's previous writing style.

Each email is:

1. Converted into an embedding
2. Stored in Pinecone
3. Associated with the current session namespace
4. Retrieved later using semantic similarity

These references are used during the refinement stage to preserve the user's
tone and writing style.

### 2. Dataset

The dataset contains external information that can be used when generating
responses.

Supported input formats include:

- PDF
- CSV

The dataset pipeline:

```text
Upload File
     │
     ▼
Extract Text
     │
     ▼
Split into Chunks
     │
     ▼
Generate Embeddings
     │
     ▼
Store Vectors in Pinecone
     │
     ▼
Session Namespace
