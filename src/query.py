from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.base_query_engine import BaseQueryEngine


def rag_query(input_dir, query) -> BaseQueryEngine:
    """Build rag pipeline on source documents and execute query."""
    reader = SimpleDirectoryReader(input_dir)
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine().query(query)


response = rag_query(
    "data/autogen", "What is conversation programming and how does autogen use it?"
)

print(response)
