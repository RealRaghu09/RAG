# Retrieval Pipeline Framework

A modular retrieval pipeline built with Python and LangChain for loading documents, chunking text, generating embeddings, indexing content with FAISS, retrieving relevant context, and applying basic input/output guardrails.

The project is designed to provide reusable building blocks for Retrieval-Augmented Generation (RAG) systems.

## Project Structure

```text
.
├── DocumentLoaders/
├── Embeddings/
├── Models/
├── chunking/
├── corpus/
├── guardrails/
├── index/
├── prompts/
├── requirements.txt
└── .gitignore
```

## Folder Overview

### DocumentLoaders

Contains utilities for loading and preprocessing documents from different sources.

Currently supported formats:

* PDF files using `PyPDFLoader`
* CSV files using `CSVLoader`

Features:

* Standard loading
* Lazy loading for memory efficiency
* Optional text splitting during loading
* Metadata extraction
* File path validation
* Unified wrapper interface through `load_documents()`

Example output:

```python
[
    (metadata, page_content),
    (metadata, page_content)
]
```

### Embeddings

Contains the embedding pipeline used to convert text into dense vector representations.

The `Embedder` class:

* Loads transformer-based embedding models from Hugging Face
* Automatically selects GPU when available
* Generates sentence embeddings using mean pooling
* Applies L2 normalization for similarity search

Default model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Output:

```python
Tensor[num_documents, embedding_dimension]
```

### Models

Contains Pydantic schemas used throughout the application.

Available schemas:

* `Chunk` — configuration for fixed-size chunking
* `overlappingChunk` — configuration for overlapping chunks
* `messageSchema` — user query and retrieved context
* `Citation` — document and chunk references
* `AnswerSchema` — final response structure with confidence score and evidence

Using schemas ensures consistent validation and data exchange across modules.

### chunking

Contains multiple chunking strategies for preparing documents before embedding.

Implemented methods:

#### Fixed Chunking

Splits text into chunks of a fixed character size.

#### Overlapping Chunking

Creates chunks with configurable overlap to preserve context across boundaries.

#### Sentence-Aware Chunking

Splits text at sentence boundaries and groups sentences while respecting the maximum chunk size.

This approach improves semantic coherence compared to fixed-size chunking.

### corpus

Stores source documents used for building the knowledge base.

Examples include:

* PDFs
* CSV files
* Web-scraped documents
* Internal datasets

These documents are processed through the loading, chunking, embedding, and indexing pipeline.

### guardrails

Contains input and output validation logic to improve system reliability.

#### Input Guardrails

Protect against:

* Excessively long inputs
* Prompt injection attempts
* Requests targeting system or developer instructions

#### Output Guardrails

Validate:

* Minimum confidence thresholds
* Empty or invalid responses

Guardrails are designed to fail fast when unsafe or low-quality interactions are detected.

### index

Contains indexing and retrieval logic.

Core components:

#### Index Builder

Responsible for:

* Generating embeddings
* Creating FAISS indexes
* Storing metadata mappings

#### Search Retriever

Provides:

* Vector similarity search
* Metadata lookup
* Document ranking
* Cross-encoder reranking

The retrieval flow is:

```text
Query
  → Embedding Generation
  → FAISS Search
  → Metadata Mapping
  → Document Ranking
  → Cross-Encoder Reranking
```

### prompts

Stores prompt templates used by downstream language models.

Typical prompt types include:

* Query rewriting
* Context formatting
* Answer generation
* Citation generation

Keeping prompts separate from application logic simplifies experimentation and versioning.

## Core Dependencies

| Package                 | Purpose                               |
| ----------------------- | ------------------------------------- |
| `langchain-community`   | Document loading utilities            |
| `transformers`          | Pretrained embedding models           |
| `torch`                 | Model inference and tensor operations |
| `sentence-transformers` | Cross-encoder reranking               |
| `faiss`                 | Vector similarity search              |
| `pydantic`              | Data validation and schemas           |
| `numpy`                 | Numerical computations                |
| `PyPDFLoader`           | PDF document processing               |
| `CSVLoader`             | CSV document processing               |

## Pipeline Overview

```text
Documents
    ↓
Document Loaders
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS Indexing
    ↓
Retrieval
    ↓
Reranking
    ↓
Guardrails
    ↓
Final Response
```



## Future Improvements

* Add hybrid search with keyword and vector retrieval

* Add evaluation metrics for retrieval quality

* Enable incremental index updates

