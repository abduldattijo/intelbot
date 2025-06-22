"""
Intelligence Document Analyzer Backend - FINAL, STABLE & SIMPLIFIED DB VERSION
"""
import os, uuid, time, logging, json, pickle, sqlite3, asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import PyPDF2, docx, numpy as np, faiss, openai, tiktoken
from sentence_transformers import SentenceTransformer

# --- Configuration & Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Definition ---
app = FastAPI(title="Intelligence Document Analyzer API", version="6.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    k: int = 5


class QueryResponse(BaseModel):
    response: str
    sources: List[Dict]


# --- Core Application Classes ---
class DocumentStore:
    """Manages document storage with a single, simplified table."""

    def __init__(self, db_path: str = "documents.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.init_database()
        logger.info(f"Database connection established to {self.db_path}")

    def init_database(self):
        cursor = self._conn.cursor()
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS chunks
                       (
                           embedding_id
                           INTEGER
                           PRIMARY
                           KEY,
                           doc_id
                           TEXT
                           NOT
                           NULL,
                           filename
                           TEXT
                           NOT
                           NULL,
                           chunk_text
                           TEXT
                           NOT
                           NULL,
                           chunk_index
                           INTEGER
                           NOT
                           NULL
                       )''')
        self._conn.commit()
        cursor.close()

    def store_chunks(self, chunks: List[Dict]):
        cursor = self._conn.cursor()
        for chunk in chunks:
            cursor.execute(
                'INSERT INTO chunks (embedding_id, doc_id, filename, chunk_text, chunk_index) VALUES (?, ?, ?, ?, ?)',
                (
                    chunk.get('embedding_id'), chunk.get('doc_id'), chunk.get('filename'),
                    chunk.get('text'), chunk.get('chunk_index')
                )
            )
        self._conn.commit()
        cursor.close()
        logger.info(f"Successfully committed {len(chunks)} chunks to the database.")

    def get_chunk_by_embedding_id(self, embedding_id: int) -> Optional[Dict]:
        cursor = self._conn.cursor()
        cursor.execute('SELECT filename, chunk_text FROM chunks WHERE embedding_id = ?', (embedding_id,))
        result = cursor.fetchone()
        cursor.close()
        if result:
            return {'filename': result[0], 'text': result[1]}
        return None


class RAGSystem:
    """The core RAG engine."""

    def __init__(self, openai_api_key: str, store: DocumentStore):
        openai.api_key = openai_api_key
        self.store = store
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index_path = "faiss_index.bin"
        self.index = self.load_or_create_index()
        logger.info("RAG system initialized.")

    def load_or_create_index(self):
        if os.path.exists(self.index_path):
            logger.info("Loading existing FAISS index...")
            return faiss.read_index(self.index_path)
        logger.info("Creating new FAISS index...")
        return faiss.IndexFlatIP(384)

    def add_document(self, doc_id: str, filename: str, content: str):
        logger.info(f"Processing and chunking {filename}...")
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = tokenizer.encode(content)
        chunks, i, max_tokens, overlap = [], 0, 300, 50
        while i < len(tokens):
            chunk_tokens = tokens[i: i + max_tokens]
            chunks.append(tokenizer.decode(chunk_tokens))
            i += max_tokens - overlap

        if not chunks:
            logger.warning(f"No text chunks generated for {filename}.")
            return

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(chunks)
        faiss.normalize_L2(embeddings)

        start_id = self.index.ntotal
        self.index.add(embeddings.astype('float32'))
        faiss.write_index(self.index, self.index_path)

        chunk_data = [{
            'embedding_id': start_id + i, 'doc_id': doc_id, 'filename': filename,
            'text': text, 'chunk_index': i
        } for i, text in enumerate(chunks)]

        self.store.store_chunks(chunk_data)
        logger.info(f"Document {filename} successfully indexed.")

    def query(self, query_text: str, k: int) -> Dict:
        logger.info("Performing similarity search...")
        query_embedding = self.embedding_model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        similarities, indices = self.index.search(query_embedding.astype('float32'), k)

        retrieved = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            chunk = self.store.get_chunk_by_embedding_id(int(idx))
            if chunk:
                retrieved.append(chunk)
            else:
                logger.error(f"FATAL: Could not find chunk for index {idx} in DB. This should not happen.")

        if not retrieved:
            return {"response": "No relevant information found.", "sources": []}

        context = "\n\n---\n\n".join([f"Source: {c['filename']}\n{c['text']}" for c in retrieved])
        system_prompt = "You are an AI intelligence analyst. Based ONLY on the provided sources, answer the user's question."
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query_text}\n\nANSWER:"

        logger.info("Generating response with OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )

        return {
            "response": response.choices[0].message.content,
            "sources": [{"filename": c['filename']} for c in retrieved]
        }


# --- FastAPI App Setup ---
@app.on_event("startup")
def on_startup():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: raise RuntimeError("OPENAI_API_KEY not set")
    app.state.store = DocumentStore()
    app.state.rag_system = RAGSystem(api_key, app.state.store)


def extract_text(file: UploadFile) -> str:
    ext = file.filename.lower().split('.')[-1]
    content = file.file.read()
    if ext == 'pdf': return "".join(page.extract_text() for page in PyPDF2.PdfReader(BytesIO(content)).pages)
    if ext == 'docx': return "\n".join(p.text for p in docx.Document(BytesIO(content)).paragraphs)
    if ext == 'txt': return content.decode('utf-8', errors='ignore')
    raise HTTPException(400, "Unsupported file type")


@app.post("/upload-document")
async def handle_upload(file: UploadFile = File(...)):
    try:
        text = extract_text(file)
        if not text.strip(): raise HTTPException(400, "Document is empty or could not be read.")
        doc_id = str(uuid.uuid4())
        app.state.rag_system.add_document(doc_id, file.filename, text)
        return {"status": "success", "filename": file.filename, "doc_id": doc_id}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return HTTPException(500, f"An error occurred: {e}")


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    result = app.state.rag_system.query(request.query, request.k)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)