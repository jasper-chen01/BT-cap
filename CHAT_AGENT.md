# AI Chat Agent Guide

The BAT Portal includes an AI-powered chat agent interface, similar to Owkin's conversational AI, that allows you to interact with the annotation system through natural language.

## Features

- **Natural Language Interaction**: Chat with the AI agent using plain English
- **File Upload via Chat**: Drag and drop files directly into the conversation
- **Intelligent Responses**: The agent understands context and provides helpful suggestions
- **Conversation History**: Maintains context throughout your session
- **Auto-Annotation**: Automatically processes files when appropriate

## Getting Started

1. Navigate to `http://localhost:8080/chat.html`
2. The agent will greet you and explain its capabilities
3. Start chatting or upload a file

## Example Conversations

### Basic Annotation
```
You: Upload my glioma data
Agent: File uploaded successfully! Ready to annotate?

You: Yes, annotate it
Agent: ✅ Successfully annotated 5,234 cells!
    - Astrocyte: 1,234 cells (23.6%)
    - Neuron: 987 cells (18.9%)
    ...
```

### Custom Parameters
```
You: Analyze with top k 20 and threshold 0.8
Agent: Processing with top_k=20, threshold=0.8...
```

### Getting Help
```
You: How does this work?
Agent: 📚 How to use the Brain Tumor Annotation Portal:
    1. Upload Data: Drag and drop a .h5ad file...
```

### Status Check
```
You: What's the system status?
Agent: ✅ System Status: Ready
    - Reference data: ✅ Loaded
    - Embeddings index: ✅ Ready (10,000 reference cells)
```

## Supported Commands

### File Operations
- "Upload file" / "Upload my data"
- "Upload [filename].h5ad"
- Drag and drop files into chat

### Annotation
- "Annotate" / "Analyze" / "Process"
- "Annotate file 1"
- "Annotate with top k 15"
- "Analyze with threshold 0.75"

### Information
- "Help" / "How does this work?"
- "What can you do?"
- "Status" / "System status"

### Parameters
The agent understands natural language parameter specification:
- "top k 20" or "top_k: 20"
- "threshold 0.8" or "similarity threshold: 0.8"

## Chat Interface Features

### Message Types
- **User Messages**: Your questions and commands (right-aligned, purple)
- **Agent Messages**: AI responses (left-aligned, white with border)
- **File Indicators**: Shows when files are uploaded
- **Annotation Results**: Displays summary statistics inline

### Suggestions
The agent provides clickable suggestion chips for:
- Quick actions (e.g., "Annotate this file")
- Common queries (e.g., "How does this work?")
- Next steps based on context

### Typing Indicator
Shows when the agent is processing your request

## API Usage

### Create Session
```python
import requests

response = requests.post('http://localhost:8000/api/chat/session')
session = response.json()
session_id = session['session_id']
```

### Send Message
```python
form_data = {
    'message': 'Upload my data',
    'file': open('data.h5ad', 'rb')
}
response = requests.post(
    f'http://localhost:8000/api/chat/{session_id}/message',
    files=form_data
)
result = response.json()
print(result['message']['content'])
```

### Annotate via Chat
```python
form_data = {
    'file_index': 0,
    'top_k': 10,
    'similarity_threshold': 0.7
}
response = requests.post(
    f'http://localhost:8000/api/chat/{session_id}/annotate',
    data=form_data
)
```

## Tips

1. **Be Natural**: The agent understands natural language, so you don't need specific commands
2. **Use Suggestions**: Click suggestion chips for quick actions
3. **Check Status**: Ask "status" if something seems wrong
4. **Multiple Files**: Upload multiple files and reference them by number ("annotate file 2")
5. **Context Aware**: The agent remembers your conversation, so you can refer back to previous messages

## Troubleshooting

### Agent doesn't understand my request
- Try rephrasing your question
- Use the suggestion chips for common actions
- Check the help message: "How does this work?"

### File upload fails
- Ensure file is in `.h5ad` format
- Check file size (very large files may timeout)
- Try uploading again

### Annotation errors
- Verify reference embeddings are loaded (ask "status")
- Check that your file contains valid single-cell data
- Ensure embeddings are present in your data

## Architecture

The chat agent uses:
- **Natural Language Processing**: Pattern matching and keyword detection
- **Context Management**: Maintains conversation history and file references
- **Intent Recognition**: Understands user goals (upload, annotate, help, etc.)
- **Parameter Extraction**: Parses parameters from natural language

Future enhancements may include:
- Machine learning-based intent classification
- More sophisticated NLP for better understanding
- Multi-turn conversation handling
- Integration with external LLMs for enhanced responses
