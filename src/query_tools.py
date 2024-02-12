from src.query import rag_query, summarization_query


def summarize_metagpt(query: str) -> str:
    """Use this function to summarize the MetaGPT paper."""
    return summarization_query("data/metagpt/MetaGPT.pdf", query)


def summarize_autogen(query: str) -> str:
    """Use this function to summarize the AutoGen paper."""
    return summarization_query("data/autogen/AutoGen.pdf", query)


def rag_metagpt(query: str) -> str:
    """Use this function for non-summarization queries on the MetaGPT paper."""
    return rag_query("data/metagpt", query)


def rag_autogen(query: str) -> str:
    """Use this function for non-summarization queries on the AutoGen paper."""
    return rag_query("data/autogen", query)


def rag_metagpt_and_autogen(query: str) -> str:
    """Use this function for non-summarization queries on both the MetaGPT and AutoGen papers."""
    return rag_query("data", query)
