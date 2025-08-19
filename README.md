# RAGPDF-Chat

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![RAG](https://img.shields.io/badge/Architecture-RAG-orange)

RAGPDF-Chat is an intelligent question-answering system that uses Retrieval-Augmented Generation (RAG) to provide precise answers from your PDF documents. Upload documents and ask natural questions about their content.

![Document Chat Demo](https://via.placeholder.com/800x400.png?text=Document+Chat+Interface+Demo)


## RAGPDFCHAT 
uses RAG architecture to extract text from a PDF chunk and store it as vectors and retrieve them to provide a well generated answer as an AI Chatbot (GPT2) 

- **PDF Text Extraction**: Accurate text extraction from PDF documents using PyMuPDF
- **Semantic Search**: Finds relevant content using sentence transformers and cosine similarity
- **Intelligent Q&A**: Answers questions using GPT-2 with context from your documents
- **Persistent Memory**: ChromaDB vector database maintains knowledge between sessions
- **Advanced NLP**: Lemmatization, query expansion, and contextual understanding
- **Interactive CLI**: Easy-to-use command-line interface
## Installation
**Clone & Setup**
python -m venv docchat_env
source docchat_env/bin/activate
git clone <your-repo-url>
cd document-chat
**Read the requirements file for further info**
pip install -r requirements.txt
**Done**
## Basic ideas 
📖 Uploaded: api_documentation.pdf

Q: How do I authenticate with the API?
A: Authentication requires an API key sent in the Authorization header as 
   "Bearer YOUR_API_KEY". Rate limits apply to all authenticated requests.

Q: What error codes might I encounter?
A: Common error codes include 400 (Bad Request), 401 (Unauthorized), 
   403 (Forbidden), and 429 (Too Many Requests).

   **or**

   📖 Uploaded: harry_potter.pdf

Q: Who is Voldemort?
A: Voldemort, also known as He-Who-Must-Not-Be-Named, is the dark wizard 
   who murdered Harry Potter's parents and seeks to conquer the wizarding world.

Q: What are Hermione's main traits?
A: Hermione Granger is known for her intelligence, dedication to learning, 
   and loyalty to her friends Harry and Ron.

 ## Technical Specifications

| Component | Model/Technology | Version | Size | Purpose |
|-----------|------------------|---------|------|---------|
| **Embedding Model** | all-MiniLM-L6-v2 | v2.2.2 | 90MB | Text vectorization |
| **Language Model** | GPT-2 | 4.30.0 | 548MB | Response generation |
| **Text Processing** | NLTK WordNet | 3.8.1 | 35MB | Lemmatization |
| **Vector Database** | ChromaDB | 0.4.15 | Varies | Semantic search |
| **PDF Processing** | PyMuPDF | 1.23.8 | 15MB | Text extraction |

## Database Specifications

| Metric | Value | Notes |
|--------|-------|-------|
| **Vector Dimension** | 384 | all-MiniLM-L6-v2 output |
| **Storage Format** | HNSW | Hierarchical Navigable Small World |
| **Distance Metric** | Cosine Similarity | Range: 0.0-1.0 |
| **Indexing Speed** | ~1000 vectors/sec | CPU-bound |
| **Query Speed** | ~5ms/query | With index |

**NOTE:** This current version does not support OCR capabilities and can process text only

**Thank you** this project is open sourced under MIT Lisence
