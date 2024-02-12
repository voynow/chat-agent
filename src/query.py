import fitz
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.base_query_engine import BaseQueryEngine
from llama_index.llms import OpenAI


def rag_query(input_dir, query) -> BaseQueryEngine:
    """Build rag pipeline on source documents and execute query."""
    reader = SimpleDirectoryReader(input_dir)
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine().query(query)


def summarization_query(pdf_path, query) -> str:
    """Send entire document to query engine for summarization."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return OpenAI().complete(query)


response = summarization_query(
    "data/autogen/AutoGen.pdf", "Summarize the autogen paper."
)

print(response)
