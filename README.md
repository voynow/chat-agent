# chat-agent
Dynamically selecting retrieval resources using GPT function calling tools


### setup environment
```bash
virtuelenv venv

source venv/bin/activate

pip install -r requirements.txt

export OPENAI_API_KEY=your-api-key
```

### Run dev tests
```bash
pip install -r requirements.dev.txt

pytest
```

### Test queries
Summarize the metagpt paper.<br>
Summarize the autogen paper.<br>
Compare and contrast how metagpt and autogen handle roles.<br>
What is conversation programming and how does autogen use it?<br>
What communication protocols are best for metagpt?<br>