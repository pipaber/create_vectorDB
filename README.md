Vector DB Pipeline
==================

This project builds and queries a LangGraph knowledge base backed by a Postgres/pgvector database. Markdown documentation in `./documents` is embedded with OpenAI models and stored in pgvector so LangChain agents can retrieve relevant context at runtime.

Features
--------
- Docker Compose recipe for a local pgvector instance (`compose.yaml`).
- Markdown ingestion pipeline (`create_vectorDB.py`) with header-aware chunking and OpenAI embeddings.
- Ready-to-use retriever tool wired to LangChain / LangGraph primitives (`test_docs.py`).

Prerequisites
-------------
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) or pip for dependency management
- Docker with Compose plugin (for the pgvector container)
- OpenAI API key with access to `text-embedding-3-large` and GPT‑4 models

Setup
-----
1. **Install dependencies**
   ```bash
   uv sync
   ```
   *(Alternative: create a virtualenv and run `pip install .`.)*

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   # edit .env with your OpenAI key and database connection string
   ```

3. **Start the vector database**
   ```bash
   docker compose up -d
   ```

4. **Populate the vector store**
   ```bash
   uv run python create_vectorDB.py
   ```
   The script splits Markdown files in `./documents`, generates embeddings, and writes them to the `PGVECTOR_COLLECTION` specified in `.env`.

5. **Smoke-test retrieval (optional)**
   ```bash
   uv run python test_docs.py
   ```
   This prints sample retrieval results for the query `"How to build a react agent?"`.

Project Structure
-----------------
- `documents/` – Markdown knowledge base seeded into pgvector.
- `create_vectorDB.py` – Ingestion pipeline for turning Markdown into embeddings.
- `test_docs.py` – Minimal LangGraph wiring to exercise the retriever tool.
- `compose.yaml` – Docker Compose definition for the pgvector database.
- `.env.example` – Template for required environment variables.
- `pyproject.toml` / `uv.lock` – Project metadata and locked dependency versions.

Next Steps
----------
- Replace the sample Markdown files with your own corpus.
- Extend the retrieval logic to power agents, chains, or API endpoints.
- Add tests or notebooks to automate ingestion and evaluation.
