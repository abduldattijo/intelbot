# backend/rag_system.py - RAG System Implementation

import os
import json
import numpy as np
import faiss
import openai
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import pickle
import tiktoken
from sentence_transformers import SentenceTransformer
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentStore:
    """Manages document storage and metadata"""

    def __init__(self, db_path: str = "documents.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for document metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS documents
                       (
                           id
                           TEXT
                           PRIMARY
                           KEY,
                           filename
                           TEXT,
                           content
                           TEXT,
                           metadata
                           TEXT,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS document_chunks
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           document_id
                           TEXT,
                           chunk_text
                           TEXT,
                           chunk_index
                           INTEGER,
                           embedding_id
                           INTEGER,
                           metadata
                           TEXT,
                           FOREIGN
                           KEY
                       (
                           document_id
                       ) REFERENCES documents
                       (
                           id
                       )
                           )
                       ''')

        conn.commit()
        conn.close()

    def store_document(self, doc_id: str, filename: str, content: str, metadata: Dict):
        """Store document and return the document ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO documents (id, filename, content, metadata)
            VALUES (?, ?, ?, ?)
        ''', (doc_id, filename, content, json.dumps(metadata)))

        conn.commit()
        conn.close()
        return doc_id

    def store_chunks(self, doc_id: str, chunks: List[Dict]):
        """Store document chunks with metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for i, chunk in enumerate(chunks):
            cursor.execute('''
                           INSERT INTO document_chunks
                               (document_id, chunk_text, chunk_index, embedding_id, metadata)
                           VALUES (?, ?, ?, ?, ?)
                           ''', (
                               doc_id,
                               chunk['text'],
                               i,
                               chunk.get('embedding_id'),
                               json.dumps(chunk.get('metadata', {}))
                           ))

        conn.commit()
        conn.close()

    def get_chunk_by_embedding_id(self, embedding_id: int) -> Optional[Dict]:
        """Retrieve chunk by embedding ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT dc.chunk_text, dc.metadata, d.filename, d.metadata as doc_metadata
                       FROM document_chunks dc
                                JOIN documents d ON dc.document_id = d.id
                       WHERE dc.embedding_id = ?
                       ''', (embedding_id,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                'text': result[0],
                'metadata': json.loads(result[1]),
                'filename': result[2],
                'document_metadata': json.loads(result[3])
            }
        return None


class IntelligenceRAG:
    """Main RAG system for intelligence document analysis"""

    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key

        # Initialize components
        self.document_store = DocumentStore()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

        # Vector database
        self.index_path = "faiss_index.bin"
        self.metadata_path = "faiss_metadata.pkl"
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension

        self.load_or_create_index()

    def load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            logger.info("Loading existing FAISS index...")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.index_metadata = pickle.load(f)
        else:
            logger.info("Creating new FAISS index...")
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            self.index_metadata = []
            self.save_index()

    def save_index(self):
        """Save FAISS index and metadata to disk"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.index_metadata, f)

    def chunk_text(self, text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        tokens = self.tokenizer.encode(text)
        chunks = []

        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            if i + max_tokens >= len(tokens):
                break

        return chunks

    def add_document(self, doc_id: str, filename: str, content: str, metadata: Dict) -> int:
        """Add document to the RAG system"""
        logger.info(f"Adding document: {filename}")

        # Store document
        self.document_store.store_document(doc_id, filename, content, metadata)

        # Chunk the document
        chunks = self.chunk_text(content)
        logger.info(f"Created {len(chunks)} chunks for {filename}")

        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to FAISS index
        start_id = self.index.ntotal
        self.index.add(embeddings.astype('float32'))

        # Store chunk metadata
        chunk_data = []
        for i, chunk in enumerate(chunks):
            embedding_id = start_id + i
            chunk_metadata = {
                'text': chunk,
                'embedding_id': embedding_id,
                'metadata': {
                    'document_id': doc_id,
                    'filename': filename,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            }
            chunk_data.append(chunk_metadata)
            self.index_metadata.append({
                'document_id': doc_id,
                'filename': filename,
                'chunk_index': i
            })

        # Store chunks in database
        self.document_store.store_chunks(doc_id, chunk_data)

        # Save updated index
        self.save_index()

        logger.info(f"Successfully added {len(chunks)} chunks to index")
        return len(chunks)

    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve most relevant chunks for a query"""
        if self.index.ntotal == 0:
            logger.warning("No documents in index")
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search for similar chunks
        similarities, indices = self.index.search(query_embedding.astype('float32'), k)

        relevant_chunks = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == -1:  # No more results
                break

            chunk_data = self.document_store.get_chunk_by_embedding_id(idx)
            if chunk_data:
                chunk_data['similarity'] = float(similarity)
                chunk_data['rank'] = i + 1
                relevant_chunks.append(chunk_data)

        return relevant_chunks

    def generate_response(self, query: str, relevant_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate response using OpenAI GPT with retrieved chunks"""

        # Prepare context from relevant chunks
        context_parts = []
        sources = []

        for chunk in relevant_chunks:
            context_parts.append(f"[Source: {chunk['filename']}]\n{chunk['text']}")
            sources.append({
                'filename': chunk['filename'],
                'similarity': chunk['similarity'],
                'rank': chunk['rank']
            })

        context = "\n\n---\n\n".join(context_parts)

        # Prepare system prompt for intelligence analysis
        system_prompt = """You are an AI intelligence analyst specializing in security and threat assessment. 
        Your task is to analyze intelligence documents and provide comprehensive, accurate responses to queries.

        Guidelines:
        - Focus on security-relevant information including threats, locations, entities, and patterns
        - Provide specific details with confidence levels where appropriate
        - Highlight geographic intelligence and temporal patterns
        - Identify potential risks and recommend actions
        - Be precise and avoid speculation beyond what's supported by the documents
        - Structure responses clearly with headers and bullet points when helpful
        """

        # Prepare user prompt
        user_prompt = f"""Based on the following intelligence documents, please answer this query: {query}

        Available Intelligence:
        {context}

        Please provide a comprehensive analysis addressing the query. Include:
        1. Direct answer to the question
        2. Supporting evidence from the documents
        3. Geographic and temporal context if relevant
        4. Security implications and recommendations
        5. Confidence level in your assessment
        """

        try:
            # Generate response using OpenAI
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.3,  # Lower temperature for more factual responses
                top_p=0.9
            )

            generated_text = response.choices[0].message.content

            return {
                "response": generated_text,
                "sources": sources,
                "query": query,
                "context_chunks": len(relevant_chunks),
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-3.5-turbo-16k"
            }

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": f"Error generating response: {str(e)}",
                "sources": sources,
                "query": query,
                "error": True
            }

    def query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Main query method - retrieve and generate"""
        logger.info(f"Processing query: {query}")

        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query, k)

        if not relevant_chunks:
            return {
                "response": "I don't have enough information in the current document database to answer your question. Please upload more relevant intelligence documents.",
                "sources": [],
                "query": query,
                "no_results": True
            }

        # Generate response
        result = self.generate_response(query, relevant_chunks)

        logger.info(f"Generated response with {len(relevant_chunks)} source chunks")
        return result

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        return {
            "total_chunks": self.index.ntotal,
            "total_documents": len(set(meta['document_id'] for meta in self.index_metadata)),
            "index_dimension": self.dimension,
            "model_name": "all-MiniLM-L6-v2"
        }


# Initialize global RAG system
_rag_system = None


def get_rag_system() -> IntelligenceRAG:
    """Get or create the global RAG system instance"""
    global _rag_system
    if _rag_system is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _rag_system = IntelligenceRAG(api_key)
    return _rag_system