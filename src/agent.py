from typing import Callable

import query_tools
from openai import OpenAI

SUMMARIZATION_FUNC_MAP = {
    "summarize_metagpt": query_tools.summarize_metagpt,
    "summarize_autogen": query_tools.summarize_autogen,
}

RAG_FUNC_MAP = {
    "rag_metagpt": query_tools.rag_metagpt,
    "rag_autogen": query_tools.rag_autogen,
    "rag_metagpt_and_autogen": query_tools.rag_metagpt_and_autogen,
}

SYS_MESSAGE = "You are a helpful and intelligent assistant. You are required to select a tool to use for the following query."


client = OpenAI()


def tool_builder(func_map: dict[str, Callable]) -> list[dict]:
    """
    Build tools object required completions API call
    https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
    """
    tools = []
    for func in func_map.values():
        tools.append(
            {
                "type": "function",
                "function": {
                    "description": func.__doc__,
                    "name": func.__name__,
                },
            }
        )
    return tools


def select_tool(query: str, func_map: dict[str, Callable]):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": SYS_MESSAGE},
            {"role": "user", "content": query},
        ],
        tools=tool_builder(func_map),
    )

    # This needs to be validated
    function_name = response.choices[0].message.tool_calls[0].function.name
    return function_name


response = select_tool(
    "What is conversation programming and how does autogen use it?", RAG_FUNC_MAP
)
print(response)
