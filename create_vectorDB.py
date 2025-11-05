# %%
import os
import sys
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

# %%
def get_markdown_files(directory):
    """Gets all markdown files in a directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".md")]

def load_and_chunk_markdown_files(file_paths):
    """Loads and chunks markdown files using a two-step process."""
    all_splits = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # First pass: split by markdown headers
        headers_to_split_on = [
            ("##", "Header_2"),
            ("###", "Header_3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text(content)

        # Second pass: split by character count
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        splits = text_splitter.split_documents(md_header_splits)

        # Add source metadata to each split
        for split in splits:
            split.metadata['source'] = file_path

        all_splits.extend(splits)
    return all_splits

def create_vector_store(documents, embeddings, connection, collection_name):
    """Creates the PGVector store."""
    return PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        connection=connection,
        collection_name=collection_name,
        use_jsonb=True,
    )

def main():
    """Main function to create the vector database from markdown files."""
    # --- Configuration ---
    connection = os.getenv("PGVECTOR_CONN")
    collection_name = os.getenv("PGVECTOR_COLLECTION")
    markdown_directory = os.getenv("DOCUMENTS_DIR", "documents")

    if not connection or not collection_name:
        print("Missing PGVECTOR_CONN or PGVECTOR_COLLECTION in environment. Check your .env.", file=sys.stderr)
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # --- 1. Load and Chunk Markdown Files ---
    print(f"Loading markdown files from: {markdown_directory}")
    markdown_files = get_markdown_files(markdown_directory)
    if not markdown_files:
        print(f"No markdown files found in '{markdown_directory}'. Please run inspect_docling_output.py first.")
        return

    doc_splits = load_and_chunk_markdown_files(markdown_files)
    print(f"Created {len(doc_splits)} document splits.")

    # --- 2. Create Vector Store ---
    print("Creating vector store...")
    vector_store = create_vector_store(doc_splits, embeddings, connection, collection_name)
    print("Vector store created successfully.")

    # --- 3. Create and Test Retriever ---
    retriever = vector_store.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_pdf_content",
        "Search and return information from PDF documents."
    )

    # Example usage
    query = "How to build a react agent?"
    print(f"\nInvoking retriever with query: '{query}'")
    results = retriever.invoke(query)
    print(results)

    print(f"\nInvoking retriever tool with query: '{query}'")
    tool_results = retriever_tool.invoke({"query": query})
    print(tool_results)

if __name__ == "__main__":
    main()
