# rag_system_fixed.py - Complete fix for RAG system index mismatch

import os
import json
import numpy as np
import faiss
import openai
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import pickle
import tiktoken
from sentence_transformers import SentenceTransformer
import sqlite3

logger = logging.getLogger(__name__)


class DocumentStore:
    """FIXED: Manages document storage and metadata with proper index mapping"""

    def __init__(self, db_path: str = "documents.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for document metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create documents table
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

        # FIXED: Add faiss_index column to track actual FAISS position
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
                           faiss_index
                           INTEGER
                           UNIQUE,
                           metadata
                           TEXT,
                           filename
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

        # Create index on faiss_index for fast lookup
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faiss_index ON document_chunks(faiss_index)')

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

    def store_chunks(self, doc_id: str, chunks: List[Dict], start_faiss_index: int):
        """FIXED: Store document chunks with proper FAISS index mapping"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for i, chunk in enumerate(chunks):
            faiss_index = start_faiss_index + i

            cursor.execute('''
                           INSERT INTO document_chunks
                               (document_id, chunk_text, chunk_index, faiss_index, metadata, filename)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ''', (
                               doc_id,
                               chunk['text'],
                               chunk['chunk_index'],
                               faiss_index,  # Use actual FAISS index
                               json.dumps(chunk.get('metadata', {})),
                               chunk.get('metadata', {}).get('filename', f'doc_{doc_id}')
                           ))

        conn.commit()
        conn.close()

    def get_chunk_by_faiss_index(self, faiss_index: int) -> Optional[Dict]:
        """FIXED: Retrieve chunk by FAISS index instead of embedding_id"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT dc.chunk_text, dc.metadata, d.filename, d.metadata as doc_metadata
                       FROM document_chunks dc
                                JOIN documents d ON dc.document_id = d.id
                       WHERE dc.faiss_index = ?
                       ''', (faiss_index,))

        result = cursor.fetchone()
        conn.close()

        if result:
            try:
                chunk_metadata = json.loads(result[1]) if result[1] else {}
                doc_metadata = json.loads(result[3]) if len(result) > 3 and result[3] else {}
            except json.JSONDecodeError:
                chunk_metadata = {}
                doc_metadata = {}

            return {
                'text': result[0],
                'metadata': chunk_metadata,
                'filename': result[2],
                'document_metadata': doc_metadata
            }

        return None

    def get_total_chunks(self) -> int:
        """Get total number of chunks in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM document_chunks")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def rebuild_index_mapping(self):
        """FIXED: Rebuild the FAISS index mapping to ensure consistency"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all chunks ordered by creation
        cursor.execute('''
                       SELECT id, chunk_text
                       FROM document_chunks
                       ORDER BY id
                       ''')

        chunks = cursor.fetchall()

        # Update faiss_index to match actual FAISS positions
        for i, (chunk_id, _) in enumerate(chunks):
            cursor.execute('''
                           UPDATE document_chunks
                           SET faiss_index = ?
                           WHERE id = ?
                           ''', (i, chunk_id))

        conn.commit()
        conn.close()

        logger.info(f"Rebuilt index mapping for {len(chunks)} chunks")


class IntelligenceRAG:
    """FIXED: Main RAG system with proper index management"""

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
        self.dimension = 384

        self.load_or_create_index()

    def load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                logger.info("Loading existing FAISS index...")
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    self.index_metadata = pickle.load(f)

                # FIXED: Verify consistency between FAISS and database
                faiss_count = self.index.ntotal
                db_count = self.document_store.get_total_chunks()

                if faiss_count != db_count:
                    logger.warning(f"Index mismatch: FAISS={faiss_count}, DB={db_count}. Rebuilding...")
                    self.rebuild_index()
                else:
                    logger.info(f"Index loaded successfully: {faiss_count} chunks")

            except Exception as e:
                logger.error(f"Error loading index: {e}. Creating new index...")
                self.create_new_index()
        else:
            logger.info("Creating new FAISS index...")
            self.create_new_index()

    def create_new_index(self):
        """Create a fresh FAISS index"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index_metadata = []
        self.save_index()

    def rebuild_index(self):
        """FIXED: Rebuild FAISS index from database to ensure consistency"""
        logger.info("Rebuilding FAISS index from database...")

        # Create new index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index_metadata = []

        # Get all chunks from database
        conn = sqlite3.connect(self.document_store.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT dc.chunk_text, dc.document_id, dc.filename, dc.chunk_index
                       FROM document_chunks dc
                       ORDER BY dc.id
                       ''')

        chunks = cursor.fetchall()
        conn.close()

        if chunks:
            # Re-encode all chunks
            texts = [chunk[0] for chunk in chunks]
            embeddings = self.embedding_model.encode(texts)
            faiss.normalize_L2(embeddings)

            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))

            # Rebuild metadata
            for i, (text, doc_id, filename, chunk_index) in enumerate(chunks):
                self.index_metadata.append({
                    'document_id': doc_id,
                    'filename': filename,
                    'chunk_index': chunk_index
                })

            # Update database with correct FAISS indices
            self.document_store.rebuild_index_mapping()

            # Save the rebuilt index
            self.save_index()

            logger.info(f"Successfully rebuilt index with {len(chunks)} chunks")

    def save_index(self):
        """Save FAISS index and metadata to disk"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.index_metadata, f)

    def chunk_text(self, text: str, max_tokens: int = 300, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        tokens = self.tokenizer.encode(text)
        chunks = []

        if len(tokens) <= max_tokens:
            return [text]

        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            if len(chunk_text.strip()) > 50:
                chunks.append(chunk_text)

            if i + max_tokens >= len(tokens):
                break

        if not chunks and text.strip():
            chunks = [text[:2000]]

        return chunks

    def add_document(self, doc_id: str, filename: str, content: str, metadata: Dict) -> int:
        """FIXED: Add document with proper index management"""
        logger.info(f"Adding document: {filename}")

        # Store document
        self.document_store.store_document(doc_id, filename, content, metadata)

        # Chunk the document
        chunks = self.chunk_text(content)
        logger.info(f"Created {len(chunks)} chunks for {filename}")

        if not chunks:
            logger.warning(f"No chunks created for {filename}")
            return 0

        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks)
        faiss.normalize_L2(embeddings)

        # FIXED: Get current index size BEFORE adding
        start_faiss_index = self.index.ntotal

        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))

        # Prepare chunk data with proper metadata
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                'text': chunk,
                'chunk_index': i,
                'metadata': {
                    'document_id': doc_id,
                    'filename': filename,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            }
            chunk_data.append(chunk_metadata)

            # Update index metadata
            self.index_metadata.append({
                'document_id': doc_id,
                'filename': filename,
                'chunk_index': i
            })

        # FIXED: Store chunks with correct FAISS indices
        self.document_store.store_chunks(doc_id, chunk_data, start_faiss_index)

        # Save updated index
        self.save_index()

        logger.info(
            f"Successfully added {len(chunks)} chunks to index (FAISS indices {start_faiss_index}-{start_faiss_index + len(chunks) - 1})")
        return len(chunks)

    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """FIXED: Retrieve most relevant chunks with proper index lookup"""
        if self.index.ntotal == 0:
            logger.warning("No documents in index")
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search for similar chunks
        search_k = min(k * 2, self.index.ntotal)
        similarities, indices = self.index.search(query_embedding.astype('float32'), search_k)

        logger.info(f"Search results for '{query}': found {len(indices[0])} candidates")

        relevant_chunks = []
        valid_results = 0

        for i, (similarity, faiss_idx) in enumerate(zip(similarities[0], indices[0])):
            if faiss_idx == -1:  # No more results
                break

            if similarity > 0.1:  # Similarity threshold
                # FIXED: Use faiss_index instead of embedding_id
                chunk_data = self.document_store.get_chunk_by_faiss_index(int(faiss_idx))

                if chunk_data:
                    chunk_data['similarity'] = float(similarity)
                    chunk_data['rank'] = valid_results + 1
                    chunk_data['faiss_index'] = int(faiss_idx)
                    relevant_chunks.append(chunk_data)
                    valid_results += 1

                    if valid_results >= k:
                        break
                else:
                    logger.warning(f"Could not retrieve chunk data for faiss_index {faiss_idx}")
            else:
                logger.debug(f"Skipping chunk at faiss_index {faiss_idx} with low similarity {similarity:.4f}")

        logger.info(f"Successfully retrieved {len(relevant_chunks)} chunks with valid data")
        return relevant_chunks

    def generate_response(self, query: str, relevant_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate response using OpenAI GPT with retrieved chunks"""
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
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.3,
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
                "context_chunks": len(relevant_chunks),
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-3.5-turbo-16k",
                "error": True
            }

    def query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Main query method - retrieve and generate"""
        logger.info(f"Processing query: {query}")

        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query, k)
        logger.info(f"Retrieved {len(relevant_chunks)} chunks for query")

        if not relevant_chunks:
            logger.warning(f"No relevant chunks found for query: {query}")
            return {
                "response": "I don't have enough information in the current document database to answer your question. Please upload more relevant intelligence documents.",
                "sources": [],
                "query": query,
                "context_chunks": 0,
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-3.5-turbo-16k",
                "no_results": True
            }

        # Generate response
        result = self.generate_response(query, relevant_chunks)
        logger.info(f"Generated response with {len(relevant_chunks)} source chunks")
        return result

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        db_chunks = self.document_store.get_total_chunks()
        faiss_chunks = self.index.ntotal

        return {
            "total_chunks": faiss_chunks,
            "database_chunks": db_chunks,
            "index_consistent": faiss_chunks == db_chunks,
            "total_documents": len(
                set(meta['document_id'] for meta in self.index_metadata)) if self.index_metadata else 0,
            "index_dimension": self.dimension,
            "model_name": "all-MiniLM-L6-v2"
        }

    def fix_index_consistency(self):
        """Public method to fix index consistency issues"""
        logger.info("Fixing index consistency...")
        self.rebuild_index()
        return self.get_index_stats()


# Utility function to fix existing RAG system
def fix_rag_system():
    """Fix existing RAG system with index mismatch issues"""
    print("🔧 Fixing RAG System Index Issues")
    print("=" * 50)

    try:
        # Initialize with OpenAI key
        import os
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY not set")
            return False

        # Create fixed RAG system
        rag_system = IntelligenceRAG(api_key)

        # Get initial stats
        initial_stats = rag_system.get_index_stats()
        print(f"📊 Initial Stats:")
        print(f"   FAISS chunks: {initial_stats['total_chunks']}")
        print(f"   Database chunks: {initial_stats['database_chunks']}")
        print(f"   Consistent: {initial_stats['index_consistent']}")

        # Fix consistency if needed
        if not initial_stats['index_consistent']:
            print("\n🔧 Fixing index consistency...")
            fixed_stats = rag_system.fix_index_consistency()
            print(f"✅ Fixed! New stats:")
            print(f"   FAISS chunks: {fixed_stats['total_chunks']}")
            print(f"   Database chunks: {fixed_stats['database_chunks']}")
            print(f"   Consistent: {fixed_stats['index_consistent']}")
        else:
            print("✅ Index is already consistent")

        # Test query
        print("\n🔍 Testing query...")
        test_result = rag_system.query("security threat", k=3)
        print(f"   Query returned: {test_result['context_chunks']} chunks")
        print(f"   No results: {test_result.get('no_results', False)}")

        if test_result['context_chunks'] > 0:
            print("✅ RAG system is working correctly!")
        else:
            print("⚠️  No chunks returned - may need more documents")

        return True

    except Exception as e:
        print(f"❌ Error fixing RAG system: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    fix_rag_system()