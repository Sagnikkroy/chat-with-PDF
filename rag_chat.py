#!/usr/bin/env python3

import os
import fitz  # PyMuPDF
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from typing import List, Dict
import warnings
import sys
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet', quiet=True)  # Download WordNet for lemmatization
warnings.filterwarnings("ignore")

class RAGChatSystem:
    def __init__(self):
        self.chat_history = []
        self.models_loaded = False
        self.embedding_model = None
        self.tokenizer = None
        self.generator = None
        self.lemmatizer = WordNetLemmatizer()
        self.setup_models()
        self.setup_chromadb()
        
    def setup_models(self):
        """Initialize models with better parameters"""
        print("Loading AI models...")
        try:
            print("Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Loading language model...")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.generator = GPT2LMHeadModel.from_pretrained('gpt2')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.models_loaded = True
            print(" Models loaded successfully!")
        except Exception as e:
            print(f" Error loading models: {e}")
            sys.exit(1)
        
    def setup_chromadb(self):
        """Initialize ChromaDB with better configuration"""
        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.chroma_client.get_or_create_collection(
                name="pdf_documents",
                metadata={"hnsw:space": "cosine"}
            )
            print(" Document collection ready")
        except Exception as e:
            print(f" ChromaDB error: {e}")
            sys.exit(1)
            
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Improved PDF text extraction with PyMuPDF"""
        try:
            print(f"📖 Reading {os.path.basename(pdf_path)}...")
            with fitz.open(pdf_path) as doc:
                return "\n".join(page.get_text() for page in doc if page.get_text())
        except Exception as e:
            print(f" PDF error: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """Better text chunking with paragraph awareness"""
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 400:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return [chunk for chunk in chunks if len(chunk.split()) > 5]
    
    def add_document_to_db(self, text: str, filename: str) -> bool:
        """Enhanced document processing"""
        chunks = self.chunk_text(text)
        if not chunks:
            return False
        try:
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=False, convert_to_tensor=True).cpu().numpy().tolist()
            self.collection.add(embeddings=embeddings, documents=chunks, ids=[f"{filename}_{i}" for i in range(len(chunks))], metadatas=[{"source": filename} for _ in chunks])
            print(f" Added {len(chunks)} chunks from {filename}")
            return True
        except Exception as e:
            print(f" DB error: {e}")
            return False
    
    def retrieve_relevant_chunks(self, query: str) -> List[str]:
        """Enhanced retrieval with adaptive relevance, aliases, and singular/plural handling"""
        try:
            original_query = query
            query_clean = re.sub(r'[^\w\s]', ' ', query.lower()).strip()
            query_words = [self.lemmatizer.lemmatize(w) for w in re.findall(r'\w+', query_clean)]
            query_variants = set()
            for word in query_words:
                query_variants.add(word)
                if word.endswith('s'):
                    query_variants.add(word[:-1])
                else:
                    query_variants.add(word + 's')
            query_variants = list(query_variants) + [query_clean]
            subject_aliases = {"voldemort": ["you-know-who", "he-who-must-not-be-named"], "ron": ["ron weasley"], "hermione": ["hermione granger"], "granger": ["hermione granger"]}
            for word in query_words:
                if word in subject_aliases:
                    query_variants.extend(subject_aliases[word])
            
            all_chunks = []
            all_distances = []
            for variant in query_variants[:4]:
                query_embedding = self.embedding_model.encode([variant]).tolist()
                results = self.collection.query(query_embeddings=query_embedding, n_results=5, include=["documents", "distances"])
                if results['documents'] and results['documents'][0]:
                    all_chunks.extend(results['documents'][0])
                    all_distances.extend(results['distances'][0])
            
            if not all_chunks:
                return []
            
            unique_chunks = {}
            for chunk, distance in zip(all_chunks, all_distances):
                if chunk not in unique_chunks or distance < unique_chunks[chunk]:
                    unique_chunks[chunk] = distance
            
            sorted_chunks = sorted(unique_chunks.items(), key=lambda x: x[1])
            threshold = 1.5 if len(sorted_chunks) <= 2 else 1.0
            
            relevant_chunks = [chunk for chunk, distance in sorted_chunks if distance < threshold]
            if not relevant_chunks and sorted_chunks:
                relevant_chunks = [sorted_chunks[0][0]]
            
            print(f"Found {len(relevant_chunks)} relevant chunks (from {len(sorted_chunks)} total)")
            if relevant_chunks and len(sorted_chunks) > 0:
                print(f" Best match distance: {sorted_chunks[0][1]:.3f}")
            
            return relevant_chunks[:4]
            
        except Exception as e:
            print(f" Retrieval error: {e}")
            return []
    
    def extract_direct_answer(self, query: str, context_chunks: List[str]) -> str:
        """Extract relevant sentences (fallback for generation)"""
        query_lower = query.lower().strip()
        full_context = "\n".join(context_chunks)
        full_context_lower = full_context.lower()
        
        print(f"🔍 Query: '{query}'")
        print(f"📝 Context preview: {full_context[:200]}...")
        
        main_subject = None
        if re.match(r"who'?s\s+(\w+)", query_lower):
            main_subject = re.match(r"who'?s\s+(\w+)", query_lower).group(1)
        elif re.match(r"who\s+is\s+(\w+)", query_lower):
            main_subject = re.match(r"who\s+is\s+(\w+)", query_lower).group(1)
        elif re.match(r"(\w+)\s+who", query_lower):
            main_subject = re.match(r"(\w+)\s+who", query_lower).group(1)
        else:
            words = query.split()
            for word in words:
                if word[0].isupper() and len(word) > 2:
                    main_subject = word.lower()
                    break
            if not main_subject:
                query_words = [w for w in re.findall(r'\w+', query_lower) if w not in {'who', 'what', 'is', 'the', 'a'}]
                main_subject = query_words[-1] if query_words else None
        
        print(f"Looking for subject: '{main_subject}'")
        if not main_subject:
            return "I don't know."
        
        subject_patterns = [main_subject, main_subject.capitalize(), main_subject.title()]
        best_sentences = []
        
        sentences = []
        for chunk in context_chunks:
            chunk_sentences = re.split(r'(?<=[.!?])\s+', chunk)
            sentences.extend([s.strip() for s in chunk_sentences if len(s.strip()) > 5])
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            mentions_subject = any(pattern.lower() in sentence_lower for pattern in subject_patterns)
            if mentions_subject:
                query_words_set = set(re.findall(r'\w+', query_lower))
                sentence_words_set = set(re.findall(r'\w+', sentence_lower))
                overlap = len(query_words_set.intersection(sentence_words_set))
                descriptive_bonus = 1 if any(word in sentence_lower for word in ['is', 'was', 'said', 'friend', 'witch']) else 0
                total_score = overlap + descriptive_bonus
                if total_score >= 1:
                    best_sentences.append((sentence.strip(), total_score))
        
        if best_sentences:
            best_sentences.sort(key=lambda x: x[1], reverse=True)
            for sentence, score in best_sentences:
                clean_sentence = re.sub(r'^[^A-Z]*', '', sentence.strip())
                if 5 <= len(clean_sentence.split()) <= 50 and not clean_sentence.startswith('"'):
                    print(f"✅ Found direct text: {clean_sentence[:100]}...")
                    return clean_sentence
            if best_sentences:
                return best_sentences[0][0].strip()
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if (main_subject in sentence_lower and len(sentence.split()) >= 8 and
                any(word in sentence_lower for word in ['is', 'was', 'said', 'friend', 'witch'])):
                clean_sentence = sentence.strip()
                print(f" Found contextual text: {clean_sentence[:100]}...")
                return clean_sentence
        
        print(" No suitable text found in context")
        return "I don't know."
    
    def generate_response(self, query: str, context_chunks: List[str]) -> str:
        """Improved generation: Always reframe/summarize with GPT-2, with looser validation"""
        if not context_chunks:
            return "I don't know - no relevant information found."
        
        # Get raw extracted text as starting point
        direct_text = self.extract_direct_answer(query, context_chunks)
        context = direct_text if direct_text != "I don't know." else " ".join(context_chunks)
        
        # Always attempt generation if context exists
        if len(context) > 50:
            print("🔄 Reframing answer with generation...")
            # Enhanced prompt for identity and summarization
            prompt = f"Using the following context from documents, provide a concise answer to the question 'Who is {query.split()[-1]}?' in your own words, summarizing key details about their identity:\nContext: {context[:800]}\nQuestion: {query}\nAnswer:"
            
            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=600
                )
                with torch.no_grad():
                    output = self.generator.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=70,
                        temperature=0.6,
                        top_k=25,
                        top_p=0.8,
                        repetition_penalty=1.2,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                response = self.tokenizer.decode(output[0], skip_special_tokens=True).split("Answer:")[-1].strip()
                response = re.sub(r'\s+', ' ', response).strip()
                
                # Looser validation
                response_lower = response.lower()
                bad_patterns = [r'she moines', r'i am \w+', r'well here goes']
                context_words = set(re.findall(r'\w+', context.lower()))
                refined_direct = re.sub(r'^.*?(is|was)\s+', '', direct_text).strip() if direct_text != "I don't know." else ""
                if (any(re.search(p, response_lower) for p in bad_patterns) or
                    len(response.split()) < 3 or len(response.split()) > 40 or
                    (len(set(re.findall(r'\w+', response_lower)).intersection(context_words)) == 0 and len(response.split()) > 15 and not refined_direct)):
                    print(" Generated response failed validation, refining fallback...")
                    if direct_text != "I don't know.":
                        return f"{query.split()[-1].capitalize()} is {refined_direct}"
                    return "I don't know."
                
                print(f" Reframes response: {response[:100]}...")
                return response
            except Exception as e:
                print(f" Generation failed: {e}")
                if direct_text != "I don't know.":
                    refined_direct = re.sub(r'^.*?(is|was)\s+', '', direct_text).strip()
                    return f"{query.split()[-1].capitalize()} is {refined_direct}"
                return "I don't know."
        
        return "I don't know."
    
    def chat(self, query: str) -> str:
        """Enhanced chat interface with better error handling"""
        if not query.strip():
            return "Please ask a question."
        
        try:
            count = self.collection.count()
            if count == 0:
                return "No documents found. Please upload a PDF first."
            print(f" Searching through {count} document chunks...")
        except:
            return "Database error. Please restart the application."
            
        print(" Searching documents...")
        chunks = self.retrieve_relevant_chunks(query)
        print(f"Retrieved chunks: {chunks}")
        
        if not chunks:
            print(" Trying broader search...")
            try:
                results = self.collection.query(
                    query_embeddings=self.embedding_model.encode([query]).tolist(),
                    n_results=3,
                    include=["documents"]
                )
                if results['documents'] and results['documents'][0]:
                    chunks = results['documents'][0]
                    print(" Using broader search results")
            except:
                pass
        
        if not chunks:
            return "I don't know - couldn't find relevant information in the documents."
        
        print(" Generating answer...")
        response = self.generate_response(query, chunks)
        self.chat_history.append({
            "user": query,
            "assistant": response,
            "context": chunks
        })
        return response

def main():
    print("=" * 60)
    print(" DOCUMENT CHAT")
    print(" Get precise answers from your documents")
    print("=" * 60)
    
    rag = RAGChatSystem()
    
    while True:
        print("\n1. Upload PDF\n2. Ask question\n3. Show chat history\n4. Exit")
        choice = input("Choose: ").strip()
        
        if choice == "1":
            path = input("PDF path: ").strip()
            if os.path.exists(path):
                text = rag.extract_text_from_pdf(path)
                if text and rag.add_document_to_db(text, os.path.basename(path)):
                    print(" Document processed successfully!")
                else:
                    print(" Failed to process document")
            else:
                print(" File not found")
        
        elif choice == "2":
            query = input("\nYour question: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            
            if not query:
                print("Please enter a question.")
                continue
                
            print("\n" + "-" * 40)
            answer = rag.chat(query)
            print(f":] {answer}")
            print("-" * 40)
        
        elif choice == "3":
            if rag.chat_history:
                print("\n📝 Chat History:")
                for i, chat in enumerate(rag.chat_history[-5:], 1):
                    print(f"\n{i}. Q: {chat['user']}")
                    print(f"   A: {chat['assistant']}")
            else:
                print("No chat history yet.")
                
        elif choice == "4":
            break
            
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Exiting...")
    except Exception as e:
        print(f"\n Fatal error: {e}")