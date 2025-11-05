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

Solution Overview (Assignment Questions)
---------------------------------------

- Use case (AI/Data Science): We are planning to use a RAG chatbot. The bot answers developer and support questions about our Markdown knowledge base by retrieving semantically similar chunks and grounding LLM responses on those sources (with citations). Primary users: engineering, support, onboarding.

- Selected NoSQL technology: Vector database (pgvector on Postgres). Although Postgres is relational, the pgvector extension provides a vector-store capability with JSONB metadata, fulfilling the “vector NoSQL” requirement for similarity search at scale. Justification: mature ecosystem, easy local/dev via Docker, strong performance for kNN search, simple ops, and easy portability to dedicated vector DBs (e.g., Qdrant/Weaviate) if needed.

- NoSQL data model: Chunks are persisted by LangChain’s PGVector integration as rows with: document (text), embedding (vector), and metadata (JSONB). We enrich metadata with Markdown header context and file source. Example metadata JSON per chunk:
  ```json
  {
    "source": "documents/guide.md",
    "Header_2": "Getting Started",
    "Header_3": "Installation"
  }
  ```
  Collections partition embeddings by `PGVECTOR_COLLECTION`. This structure supports fast similarity search plus flexible filtering on metadata.

- Architecture and business value: Ingestion splits Markdown by headers, then into ~1k‑char chunks, embeds with OpenAI, and writes to pgvector. At query time, the chatbot retrieves top‑k similar chunks, composes a prompt with the retrieved context, and generates an answer using an LLM. Benefits: faster time‑to‑answer, consistent responses grounded in source docs, reduced support load, and improved onboarding.

Next Steps
----------
- Replace the sample Markdown files with your own corpus.
- Extend the retrieval logic to power agents, chains, or API endpoints.
- Add tests or notebooks to automate ingestion and evaluation.

RAG Chatbot with LangGraph `create_agent`
-----------------------------------------

Below is a minimal, end‑to‑end example that turns the PGVector retriever into a RAG chatbot using LangGraph’s `create_agent`. It does not require manual compilation; the returned graph supports `.invoke/.stream/.astream` directly.

Example: `chatbot_rag.py`
```python
import asyncio
import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

# Depending on your LangGraph version, the import may be either of these:
# from langgraph import create_agent
from langgraph.prebuilt import create_agent  # fallback for prebuilt entrypoints


def build_rag_agent():
    load_dotenv()

    # Vector store / retriever
    conn = os.environ["PGVECTOR_CONN"]
    collection = os.environ["PGVECTOR_COLLECTION"]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection,
        connection=conn,
        use_jsonb=True,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    rag_tool = create_retriever_tool(
        retriever,
        name="retrieve_docs",
        description=(
            "Search the Markdown knowledge base and return the most relevant excerpts. "
            "Use this to ground your answers and provide citations."
        ),
    )

    system_prompt = (
        "You are a helpful RAG chatbot. Always call `retrieve_docs` when answering. "
        "Ground responses in retrieved snippets and include a Sources section with file paths from metadata."
    )

    # `create_agent` accepts a model identifier string or a BaseChatModel.
    # Example providers: "openai:gpt-4o-mini", "anthropic:claude-3-5-sonnet-latest".
    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[rag_tool],
        system_prompt=system_prompt,
        # You can also pass checkpointer/store here if you want persistence:
        # checkpointer=AsyncPostgresSaver.from_conn_string(<conn>),
        # store=AsyncPostgresStore.from_conn_string(<conn>),
    )
    return agent


async def main():
    agent = build_rag_agent()

    print("RAG Chatbot ready. Type 'quit' to exit.\n")
    history = []  # maintain conversational context locally

    while True:
        try:
            text = input("User: ")
            if text.lower() in {"q", "quit", "exit"}:
                print("Goodbye!")
                break

            # Stream values with astream; state contains the evolving AgentState.
            # We keep the last yielded state as the final result for this turn.
            last_state = None
            async for state in agent.astream(
                {"messages": history + [HumanMessage(content=text)]},
                stream_mode="values",
            ):
                last_state = state

            if not last_state:
                print("AI: <no response>")
                continue

            messages = last_state["messages"]
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                print(f"AI: {last_msg.content}")
            else:
                content = getattr(last_msg, "content", None)
                print(f"AI: {content or '<no content>'}")

            # Carry forward the conversation state
            history = messages

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())
```

Notes
- Model provider string: Replace `"openai:gpt-4o-mini"` with your preferred model. Ensure corresponding API keys are set in `.env`.
- Connection string: `langchain-postgres` uses psycopg3. Use `postgresql+psycopg://...` for PGVector. If you later decide to add LangGraph Postgres checkpointers/stores, some helpers prefer `postgresql://...` (no driver suffix).
- Streaming: The example uses `astream(..., stream_mode="values")` as requested. For more granular updates (tool/LLM steps), use `stream_mode="updates"`.

Persistent Memory (Postgres checkpointer/store)
----------------------------------------------

To persist conversations (threads) and give the agent memory across turns and restarts, use LangGraph’s Postgres checkpointer and store. Note: these expect a `postgresql://` URL (no `+psycopg`), so we convert from `PGVECTOR_CONN` when needed.

Example: `chatbot_rag_memory.py`
```python
import asyncio
import os
import uuid
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langgraph import create_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore


def build_retriever_tool():
    conn = os.environ["PGVECTOR_CONN"]  # e.g., postgresql+psycopg://...
    collection = os.environ["PGVECTOR_COLLECTION"]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection,
        connection=conn,
        use_jsonb=True,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return create_retriever_tool(
        retriever,
        name="retrieve_docs",
        description=(
            "Search the Markdown knowledge base and return the most relevant excerpts. "
            "Use this to ground your answers and provide citations."
        ),
    )


async def main():
    load_dotenv()

    # LangGraph Postgres helpers prefer postgresql:// (no +psycopg)
    pg_conn = os.environ["PGVECTOR_CONN"]
    if pg_conn.startswith("postgresql+psycopg://"):
        pg_conn = pg_conn.replace("postgresql+psycopg://", "postgresql://", 1)

    rag_tool = build_retriever_tool()

    system_prompt = (
        "You are a helpful RAG chatbot. Always call `retrieve_docs` when needed. "
        "Ground responses in retrieved snippets and include a Sources section."
    )

    async with (
        AsyncPostgresSaver.from_conn_string(pg_conn) as checkpointer,
        AsyncPostgresStore.from_conn_string(pg_conn) as store,
    ):
        # Best-effort initialize tables
        try:
            await checkpointer.setup()
        except Exception:
            pass
        try:
            await store.setup()
        except Exception:
            pass

        agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=[rag_tool],
            system_prompt=system_prompt,
            checkpointer=checkpointer,
            store=store,
        )

        user_id = os.getenv("DEMO_USER_ID", "123456")
        thread_id = f"thread_{uuid.uuid4()}"
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

        print("RAG Chatbot with memory ready. Type 'quit' to exit.")
        print(f"Running with user_id={user_id} thread_id={thread_id}\n")

        while True:
            try:
                text = input("User: ")
                if text.lower() in {"q", "quit", "exit"}:
                    print("Goodbye!")
                    break

                # Only pass the new user message; memory is handled by checkpointer.
                last_state = None
                async for state in agent.astream(
                    {"messages": [HumanMessage(content=text)]},
                    config=config,
                    stream_mode="values",
                ):
                    last_state = state

                if not last_state:
                    print("AI: <no response>")
                    continue

                last_msg = last_state["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    print(f"AI: {last_msg.content}")
                else:
                    content = getattr(last_msg, "content", None)
                    print(f"AI: {content or '<no content>'}")

            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break


if __name__ == "__main__":
    asyncio.run(main())
```

Tips
- If `from langgraph import create_agent` fails with your version, try `from langgraph.prebuilt import create_agent`.
- Keep using `PGVECTOR_CONN` for the vector store (psycopg3 URL). Convert to `postgresql://` only for LangGraph’s checkpointer/store helpers.
