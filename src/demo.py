import agent

queries = [
    "Summarize the metagpt paper.",
    "Summarize the autogen paper.",
    "What communication protocols are best for metagpt?",
    "What is conversation programming and how does autogen use it?",
    "Compare and contrast how metagpt and autogen handle roles.",
]


for query in queries:
    print("*" * 75)
    print(f"agent.runner(query={query})")
    print("*" * 75)
    response = agent.runner(query=query)
    print(f"Agent response: {response}")
