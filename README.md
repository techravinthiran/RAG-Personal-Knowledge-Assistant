# RAG Chatbot System

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and HuggingFace Transformers. This system can answer questions based on uploaded documents using both local and cloud-based language models.

## Features

- 📄 **Document Processing**: Upload and process PDF, TXT, and DOCX files
- 🔍 **Smart Retrieval**: FAISS vector store for efficient document similarity search
- 🤖 **Multiple LLM Support**: 
  - Local: HuggingFace Flan-T5 (free, no API key required)
  - Cloud: Groq LLaMA3 (better quality, requires API key)
- 💬 **Interactive Chat**: Clean chat interface with source citations
- 🔐 **Secure API Management**: Separate configuration file for API keys

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup

### API Configuration (Optional)

For better performance with Groq LLaMA3:

1. Get a free API key from [Groq Console](https://console.groq.com)
2. Create a `config.py` file:
```python
GROQ_API_KEY = 'your-groq-api-key-here'
```

The system will work without an API key using the local Flan-T5 model.

## Usage

### Streamlit App

Run the web interface:
```bash
streamlit run app.py
```

1. Upload documents using the sidebar
2. Click "Build Knowledge Base" to process documents
3. Select your preferred model (Flan-T5 or Groq LLaMA3)
4. Start chatting with your documents!

### Jupyter Notebook

For development and testing:
```bash
jupyter notebook apps.ipynb
```

The notebook contains the complete RAG pipeline with step-by-step implementation and testing.

## Project Structure

```
Mini-Project-5/
├── app.py              # Streamlit web application
├── apps.ipynb          # Jupyter notebook with RAG pipeline
├── config.py           # API key configuration (create this)
├── requirements.txt    # Python dependencies
├── dataset/           # Sample documents
├── faiss_index/       # Vector store (created automatically)
└── README.md          # This file
```

## Technical Details

### RAG Pipeline Components

1. **Document Loading**: Support for PDF, TXT, DOCX files
2. **Text Chunking**: Recursive character splitting with overlap
3. **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
4. **Vector Store**: FAISS for similarity search
5. **LLM Integration**: Flan-T5 and Groq LLaMA3
6. **Prompt Engineering**: Context-aware question answering

### Dependencies

- **Streamlit**: Web application framework
- **LangChain**: RAG pipeline orchestration
- **FAISS**: Vector similarity search
- **Transformers**: HuggingFace model integration
- **Sentence-Transformers**: Text embeddings
- **Groq**: Cloud LLM API client
- **PyPDF2**: PDF processing
- **python-docx**: DOCX processing

## Models

### Flan-T5 (Local)
- **Model**: `google/flan-t5-base`
- **Advantages**: Free, no internet required, privacy-focused
- **Limitations**: Smaller context window, basic responses

### Groq LLaMA3 (Cloud)
- **Model**: `llama-3.1-8b-instant`
- **Advantages**: Better quality, faster inference, larger context
- **Requirements**: API key and internet connection

## Security Notes

- Add `config.py` to `.gitignore` to protect API keys
- The system does not store document content permanently
- Local processing ensures document privacy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed
2. **Invalid API Key**: Check your Groq API key in `config.py`
3. **Memory Issues**: Reduce chunk size or use smaller documents
4. **Model Loading Errors**: Check internet connection for first-time downloads

### Performance Tips

- Use GPU for embeddings if available
- Preprocess documents to optimal size
- Cache embeddings for repeated use
- Choose appropriate chunk size based on document type

## Future Enhancements

- [ ] Support for more document formats
- [ ] Conversation history persistence
- [ ] Multiple vector store backends
- [ ] Advanced chunking strategies
- [ ] Model fine-tuning capabilities
- [ ] Batch document processing
- [ ] Real-time document updates
