import os
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.tools import create_retriever_tool


def main() -> None:
    load_dotenv()

    connection = os.getenv("PGVECTOR_CONN")
    collection_name = os.getenv("PGVECTOR_COLLECTION")
    query = os.getenv("RETRIEVAL_QUERY", "How to build a react agent?")

    if not connection or not collection_name:
        print("Missing PGVECTOR_CONN or PGVECTOR_COLLECTION in environment. Check your .env.", file=sys.stderr)
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )

    retriever = vector_store.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_information",
        "Search and return information about LangGraph",
    )

    print(f"Retriever results for: {query}")
    for idx, doc in enumerate(retriever.invoke(query), start=1):
        source = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"[{idx}] {source} :: {preview}...")

    print("\nTool invocation payload:")
    print(retriever_tool.invoke({"query": query}))


if __name__ == "__main__":
    main()
