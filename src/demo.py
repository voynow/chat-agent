import agent

sep_str = "*" * 100

queries = [
    "Summarize the metagpt paper.",
    "Summarize the autogen paper.",
    "What communication protocols are best for metagpt?",
    "What is conversation programming and how does autogen use it?",
    "Compare and contrast how metagpt and autogen handle roles.",
]


for query in queries:
    print(f"{sep_str}\nagent.runner(query={query})\n{sep_str}")
    response = agent.runner(query=query)
    print(f"Agent response: {response}")
