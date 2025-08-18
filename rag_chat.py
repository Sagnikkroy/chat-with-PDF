#!/usr/bin/env python3
"""
RAG Chat Application - Working Version
Properly answers questions from uploaded documents
"""

import os
import PyPDF2
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from typing import List, Dict
import warnings
import sys
from difflib import SequenceMatcher

warnings.filterwarnings("ignore")

class RAGChatSystem:
    def __init__(self):
        self.chat_history = []
        self.models_loaded = False
        self.embedding_model = None
        self.tokenizer = None
        self.generator = None
        self.setup_models()
        self.setup_chromadb()
        
    def setup_models(self):
        """Initialize the embedding and generation models"""
        print("📥 Loading AI models (first time only - please wait)...")
        print("⏳ This may take 1-2 minutes...")
        
        try:
            # Use sentence-transformers for better embeddings
            print("🔍 Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Use GPT-2 for text generation
            print("🧠 Loading GPT-2 model...")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.generator = GPT2LMHeadModel.from_pretrained('gpt2')
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.models_loaded = True
            print("✅ Models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("💡 Try running: pip install torch transformers sentence-transformers")
            sys.exit(1)
        
    def setup_chromadb(self):
        """Initialize ChromaDB client"""
        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            
            try:
                self.collection = self.chroma_client.get_collection("pdf_documents")
                print("📚 Connected to existing document collection")
            except:
                self.collection = self.chroma_client.create_collection("pdf_documents")
                print("📚 Created new document collection")
        except Exception as e:
            print(f"❌ ChromaDB error: {e}")
            print("💡 Try running: pip install chromadb")
            
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            print(f"📖 Reading PDF: {os.path.basename(pdf_path)}")
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                total_pages = len(pdf_reader.pages)
                for i, page in enumerate(pdf_reader.pages, 1):
                    text += page.extract_text() + "\n"
                    if i % 5 == 0 or i == total_pages:
                        print(f"📄 Processed {i}/{total_pages} pages...")
                        
                print(f"✅ Extracted {len(text)} characters from {total_pages} pages")
                return text
        except Exception as e:
            print(f"❌ Error reading PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
                
        return chunks
    
    def add_document_to_db(self, text: str, filename: str) -> bool:
        """Process and add document to ChromaDB"""
        print(f"⚙️ Processing: {filename}")
        
        chunks = self.chunk_text(text)
        if not chunks:
            print("❌ No text chunks generated")
            return False
            
        print(f"📝 Created {len(chunks)} text chunks")
        print("🔍 Generating embeddings...")
        
        try:
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=False).tolist()
            ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
            
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                ids=ids,
                metadatas=[{"filename": filename, "chunk_id": i} for i in range(len(chunks))]
            )
            print(f"✅ Added {len(chunks)} chunks to database")
            return True
        except Exception as e:
            print(f"❌ Database error: {str(e)}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, n_results: int = 5) -> List[str]:
        """Retrieve relevant text chunks with better query handling"""
        try:
            processed_query = " ".join(query.lower().split())
            query_embedding = self.embedding_model.encode([processed_query]).tolist()
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=["documents", "distances"]
            )
            
            if results['distances'] and results['distances'][0]:
                max_distance = 1.2  # Increased threshold
                filtered_docs = [
                    doc for doc, dist in zip(results['documents'][0], results['distances'][0])
                    if dist < max_distance
                ]
                return filtered_docs
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            return []
    
    def answer_in_context(self, answer: str, context_chunks: List[str]) -> bool:
        """Improved answer verification"""
        answer = answer.lower().strip()
        if not answer or answer == "i don't know":
            return False
            
        answer_terms = set(re.findall(r'\w+', answer))
        for chunk in context_chunks:
            chunk_terms = set(re.findall(r'\w+', chunk.lower()))
            if len(answer_terms & chunk_terms) >= 2:  # At least 2 matching terms
                return True
        return False
    
    def generate_response(self, query: str, context_chunks: List[str]) -> str:
        """Improved generation with better prompt"""
        if not context_chunks:
            return "I don't know."
            
        context = "\n".join(context_chunks)[:1200]  # Larger context window
        
        prompt = (
            "Answer the question using ONLY the following context. "
            "If the answer isn't in the context, say 'I don't know'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=1024, truncation=True)
            
            with torch.no_grad():
                outputs = self.generator.generate(
                    inputs,
                    max_length=inputs.shape[1] + 150,
                    num_return_sequences=1,
                    temperature=0.6,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Answer:")[-1].strip()
            
            # Clean response
            response = re.sub(r'\[.*?\]', '', response)  # Remove citations
            response = response.split('\n')[0].strip()
            
            if not response or not self.answer_in_context(response, context_chunks):
                return "I don't know."
                
            return response
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return "I don't know."
    
    def chat(self, query: str) -> str:
        """Main chat function"""
        relevant_chunks = self.retrieve_relevant_chunks(query)
        response = self.generate_response(query, relevant_chunks)
        self.chat_history.append({"user": query, "assistant": response})
        return response

def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("📚 DOCUMENT CHAT ASSISTANT")
    print("💬 Get answers from your uploaded PDFs")
    print("=" * 60)

def print_menu():
    """Print main menu"""
    print("\n📋 MENU:")
    print("1. 📁 Upload PDF")
    print("2. 💬 Chat with documents")
    print("3. 📜 Show chat history")
    print("4. 📊 Show database info") 
    print("5. 🗑️ Clear chat history")
    print("6. ❌ Exit")

def get_pdf_path():
    """Get PDF path from user with validation"""
    while True:
        pdf_path = input("\n📁 Enter PDF file path (or drag & drop file here): ").strip().strip('"')
        
        if os.path.exists(pdf_path) and pdf_path.lower().endswith('.pdf'):
            return pdf_path
        elif pdf_path.lower() == 'back':
            return None
        else:
            print("❌ File not found or not a PDF. Type 'back' to return to menu.")

def main():
    """Main application loop"""
    print_banner()
    
    # Initialize system
    print("🚀 Initializing system...")
    rag_system = RAGChatSystem()
    
    print("✅ System ready!")
    
    while True:
        print_menu()
        choice = input("\n👉 Choose option (1-6): ").strip()
        
        if choice == "1":
            pdf_path = get_pdf_path()
            if pdf_path:
                print(f"\n⚙️ Processing: {os.path.basename(pdf_path)}")
                text = rag_system.extract_text_from_pdf(pdf_path)
                if text:
                    filename = os.path.basename(pdf_path)
                    if rag_system.add_document_to_db(text, filename):
                        print("🎉 PDF processed successfully!")
        
        elif choice == "2":
            try:
                count = rag_system.collection.count()
                if count == 0:
                    print("⚠️ No documents in database. Please upload a PDF first.")
                    continue
            except:
                print("⚠️ No documents in database. Please upload a PDF first.")
                continue
                
            print("\n💬 Chat Mode - Type 'back' to return to menu")
            print("-" * 40)
            print("ℹ️ Ask questions about your uploaded documents")
            
            while True:
                query = input("\n//1 Your question: ").strip()
                
                if query.lower() == 'back':
                    break
                elif query:
                    print(":0 Thinking...")
                    response = rag_system.chat(query)
                    print(f"\n :)  {response}")
                    print("-" * 40)
                else:
                    print("⚠️ Please enter a question or 'back' to return")
        
        elif choice == "3":
            print("\n💬 CHAT HISTORY:")
            print("-" * 40)
            if rag_system.chat_history:
                for i, chat in enumerate(rag_system.chat_history, 1):
                    print(f"[{i}] // You: {chat['user']}")
                    print(f"[{i}] :) Bot: {chat['assistant']}")
                    print("-" * 30)
            else:
                print("No conversations yet!")
        
        elif choice == "4":
            try:
                count = rag_system.collection.count()
                print(f"\n📊 Database contains {count} document chunks")
                
                if count > 0:
                    results = rag_system.collection.get()
                    filenames = set()
                    for metadata in results['metadatas']:
                        if 'filename' in metadata:
                            filenames.add(metadata['filename'])
                    
                    if filenames:
                        print("📄 Documents in database:")
                        for filename in filenames:
                            print(f"  • {filename}")
            except Exception as e:
                print(f"❌ Error getting database info: {e}")
        
        elif choice == "5":
            rag_system.chat_history = []
            print("🗑️ Chat history cleared!")
        
        elif choice == "6":
            print("👋 Thank you for using Document Chat Assistant! Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-6.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Application closed by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        input("Press Enter to exit...")