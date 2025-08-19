# RAGPDF-Chat

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![RAG](https://img.shields.io/badge/Architecture-RAG-orange)

Document Chat is an intelligent question-answering system that uses Retrieval-Augmented Generation (RAG) to provide precise answers from your PDF documents. Upload documents and ask natural questions about their content.

![Document Chat Demo](https://via.placeholder.com/800x400.png?text=Document+Chat+Interface+Demo)


## RAGPDFCHAT 
uses RAG architecture to extract text from a PDF chunk and store it as vectors and retrieve them to provide a well generated answer as an AI Chatbot (GPT2) 

- **PDF Text Extraction**: Accurate text extraction from PDF documents using PyMuPDF
- **Semantic Search**: Finds relevant content using sentence transformers and cosine similarity
- **Intelligent Q&A**: Answers questions using GPT-2 with context from your documents
- **Persistent Memory**: ChromaDB vector database maintains knowledge between sessions
- **Advanced NLP**: Lemmatization, query expansion, and contextual understanding
- **Interactive CLI**: Easy-to-use command-line interface
