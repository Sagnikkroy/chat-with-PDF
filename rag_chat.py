import os
import re
import PyPDF2
import torch
import chromadb
import pytesseract
from typing import List
from PIL import Image
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from chromadb.utils import embedding_functions

class RAGChatSystem:
    def __init__(self):
        self.chat_history = []
        self.show_intro_art()
        self.setup_models()
        self.setup_chromadb()
        
    def show_intro_art(self):
        """Display the ASCII art intro"""
        print("""
▀█▀ ▄▀█ █░░ █▄▀   ▀█▀ █▀█   █▀█ █▀▄ █▀▀   ▀ ▀▄
░█░ █▀█ █▄▄ █░█   ░█░ █▄█   █▀▀ █▄▀ █▀░   ▄ ▄▀
        """)
        print("with this RAG based chat System in this terminal you can talk about your pdf directly to this chatbot ENJOY")
        print("="*50)
        
    def setup_models(self):
        """Initialize the embedding and generation models"""
        print("Loading models...")
        
        # Use state-of-the-art embedding model
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        # Use Mistral 7B for text generation
        print("Loading Model... (this may take a while)")
        self.tokenizer = AutoTokenizer.from_pretrained(
            'mistralai/Mistral-7B-Instruct-v0.2',
            padding_side="left"
        )
        self.generator = AutoModelForCausalLM.from_pretrained(
            'mistralai/Mistral-7B-Instruct-v0.2',
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Model loaded successfully!")
        print(f"Embedding model: BAAI/bge-large-en-v1.5")
        print(f"LLM: Mistral-7B-Instruct-v0.2")
        print("="*50)
        
    def setup_chromadb(self):
        """Initialize ChromaDB client"""
        # Create a persistent ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection with embedding function
        try:
            self.collection = self.chroma_client.get_collection("pdf_documents")
            print("Connected to existing document collection")
        except:
            self.collection = self.chroma_client.create_collection("pdf_documents")
            print("Created new document collection")
            
    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extract text from PDF using OCR (for scanned PDFs)"""
        try:
            images = convert_from_path(pdf_path)
            text = ""
            
            for i, img in enumerate(images):
                text += pytesseract.image_to_string(img) + "\n"
                print(f"Processed page {i+1} with OCR")
                
            return text
        except Exception as e:
            print(f"OCR Error: {str(e)}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file with OCR fallback"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:  # If text was extracted normally
                        text += page_text + "\n"
                    else:  # Fallback to OCR for this page
                        print(f"⚠️ Page {page_num+1} has no text, attempting OCR...")
                        # Convert just this page to image for OCR
                        images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
                        if images:
                            text += pytesseract.image_to_string(images[0]) + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks using sentence-aware splitting"""
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split by sentences first (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Start new chunk with overlap
                    overlap_words = ' '.join(current_chunk).split()[-overlap:]
                    current_chunk = [' '.join(overlap_words)] if overlap > 0 else []
                    current_chunk.append(sentence)
                    current_length = len(' '.join(current_chunk).split())
                else:
                    # Very long sentence case
                    words = sentence.split()
                    for i in range(0, len(words), chunk_size):
                        chunks.append(' '.join(words[i:i+chunk_size]))
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def add_document_to_db(self, text: str, filename: str) -> bool:
        """Process and add document to ChromaDB"""
        print(f"Processing document: {filename}")
        
        # Chunk the text
        chunks = self.chunk_text(text)
        
        if not chunks:
            print("No text chunks generated from the PDF")
            return False
            
        print(f"Generated {len(chunks)} text chunks")
        
        # Generate embeddings in batches to avoid memory issues
        batch_size = 32
        embeddings = []
        print("Creating embeddings...")
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=True)
            embeddings.extend(batch_embeddings.tolist())
        
        # Create unique IDs for chunks
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        
        # Add to ChromaDB
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                ids=ids,
                metadatas=[{"filename": filename, "chunk_id": i} for i in range(len(chunks))]
            )
            print(f"Successfully added {len(chunks)} to database")
            return True
        except Exception as e:
            print(f"❌ Error adding document to database: {str(e)}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve relevant text chunks based on query"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            print(f"Error retrieving yo documents: {str(e)}")
            return []
    
    def generate_response(self, query: str, context_chunks: List[str]) -> str:
        """Generate response using Mistral with retrieved context"""
        # Combine context chunks
        context = "\n\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Create Mistral-style prompt
        prompt = f"""<s>[INST] You are a helpful teaching AI assistant. Answer the question using only on the provided context.

{context}

Question: {query}

Answer clearly and concisely. If you don't know the answer, say "Ask the question only from the document :\ " [/INST]"""
        
        # Tokenize and generate
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=4096  # Mistral's context window
            ).to(self.generator.device)
            
            with torch.no_grad():
                outputs = self.generator.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            answer_start = full_response.find("[/INST]") + len("[/INST]")
            response = full_response[answer_start:].strip()
            
            # Clean up any remaining special tokens
            response = response.replace("</s>", "").strip()
            
            return response if response else "I couldn't generate a response based on the provided context."
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def chat(self, query: str) -> str:
        """Main chat function that combines retrieval and generation"""
        if not query.strip():
            return "Please enter a valid question."
            
        print(f"\nSearching for: {query}")
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query)
        
        if not relevant_chunks:
            response = "I couldn't find relevant information in the uploaded documents to answer your question."
        else:
            print(f"Found {len(relevant_chunks)} relevant chunks")
            # Generate response
            response = self.generate_response(query, relevant_chunks)
        
        # Add to chat history
        self.chat_history.append({"user": query, "assistant": response})
        
        return response
    
    def get_database_info(self):
        """Get information about the current database"""
        try:
            count = self.collection.count()
            return f"Database contains {count} document chunks"
        except:
            return "Database is empty"

# Helper functions for the interface
def upload_pdf(pdf_path: str):
    """Upload and process a PDF file"""
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return False
    
    print(f"\nLoading PDF: {pdf_path}")
    
    # Extract text
    text = rag_system.extract_text_from_pdf(pdf_path)
    
    if text:
        print(f"Extracted {len(text)} characters")
        # Add to database
        filename = os.path.basename(pdf_path)
        success = rag_system.add_document_to_db(text, filename)
        return success
    else:
        print("Could not extract text from PDF")
        return False

def chat_with_pdf(query: str):
    """Chat with the uploaded PDF"""
    if not query.strip():
        print("⚠️ Please enter a question!")
        return
    
    response = rag_system.chat(query)
    
    # Display the conversation
    print("\n" + "=" * 50)
    print(f"👤// You: {query}")
    print(f":) Assistant: {response}")
    print("=" * 50 + "\n")
    
    return response

def show_chat_history():
    """Display the full chat history"""
    print("\nCHAT HISTORY")
    print("=" * 50)
    
    if not rag_system.chat_history:
        print("No conversations yet. Start chatting!")
        return
    
    for i, chat in enumerate(rag_system.chat_history, 1):
        print(f"[{i}] 👤 You: {chat['user']}")
        print(f"[{i}] 🤖 Assistant: {chat['assistant']}")
        print("-" * 50)

def clear_chat_history():
    """Clear the chat history"""
    rag_system.chat_history = []
    print("Chat history cleared!")

def show_database_info():
    """Show database information"""
    info = rag_system.get_database_info()
    print(f"\n{info}\n")

# Initialize the system
print("\nInitializing PDF RAG Chat System...")
rag_system = RAGChatSystem()

# Example usage
print("""
System Ready!
      
Commands:
1. upload_pdf("path/to/your/document.pdf") - Upload a PDF
2. chat_with_pdf("Your question") - Ask about documents
3. show_chat_history() - View conversation history
4. clear_chat_history() - Reset conversations
5. show_database_info() - Check loaded documents

Example:
upload_pdf("research_paper.pdf")
chat_with_pdf("What is the main research question?")
""")