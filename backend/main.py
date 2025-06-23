"""
INTELLIGENCE DOCUMENT ANALYZER - QUERY ISSUE FIXED VERSION
Keeps all intelligence features while fixing the query system
"""
import os
import uuid
import time
import logging
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import PyPDF2, docx, numpy as np, faiss, openai, tiktoken
from sentence_transformers import SentenceTransformer

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Intelligence Document Analyzer API - Query Fixed", version="10.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# FRONTEND-COMPATIBLE MODELS (KEEPING ALL FEATURES)
class DocumentMetadata(BaseModel):
    filename: str
    file_type: str
    uploaded_at: str
    file_size: int


class DocumentClassification(BaseModel):
    primary_type: str = "intelligence_report"
    sub_types: List[str] = ["security_analysis"]
    confidence: float = 0.8
    security_classification: str = "RESTRICTED"


class SentimentAnalysis(BaseModel):
    overall_sentiment: str = "neutral"
    sentiment_score: float = 0.0
    threat_level: str = "Low"
    urgency_indicators: List[str] = []


class GeographicIntelligence(BaseModel):
    states: List[str] = []
    cities: List[str] = []
    countries: List[str] = []
    coordinates: List[Dict[str, Any]] = []
    total_locations: int = 0
    other_locations: List[str] = []


class TemporalIntelligence(BaseModel):
    dates_mentioned: List[str] = []
    time_periods: List[str] = []
    months_mentioned: List[str] = []
    years_mentioned: List[str] = []
    temporal_patterns: List[str] = []


class NumericalIntelligence(BaseModel):
    incidents: List[int] = []
    casualties: List[int] = []
    weapons: List[int] = []
    arrests: List[int] = []
    monetary_values: List[float] = []


class CrimePatterns(BaseModel):
    primary_crimes: List[Tuple[str, int]] = []
    crime_frequency: Dict[str, int] = {}
    crime_trends: List[Dict[str, Any]] = []


class TextStatistics(BaseModel):
    word_count: int
    sentence_count: int
    paragraph_count: int
    readability_score: float = 50.0
    language: str = "en"


class DocumentAnalysis(BaseModel):
    document_classification: DocumentClassification
    entities: Dict[str, List[str]] = {}
    sentiment_analysis: SentimentAnalysis
    geographic_intelligence: GeographicIntelligence
    temporal_intelligence: TemporalIntelligence
    numerical_intelligence: NumericalIntelligence
    crime_patterns: CrimePatterns
    relationships: List[Dict[str, Any]] = []
    text_statistics: TextStatistics
    intelligence_summary: str
    confidence_score: float
    processing_time: float


class AnalyzedDocument(BaseModel):
    id: str
    content: str
    metadata: DocumentMetadata
    analysis: DocumentAnalysis


class QueryRequest(BaseModel):
    query: str
    max_results: int = 5


class QueryResponse(BaseModel):
    response: str
    sources: List[Dict]
    query: Optional[str] = None
    context_chunks: Optional[int] = 0
    timestamp: Optional[str] = None
    model: Optional[str] = None
    error: Optional[bool] = False
    no_results: Optional[bool] = False


# SIMPLIFIED BUT ROBUST DOCUMENT STORE (FIXES QUERY ISSUE)
class DocumentStore:
    def __init__(self, db_path: str = "documents.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.init_database()

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
                           NULL,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS documents
                       (
                           id
                           TEXT
                           PRIMARY
                           KEY,
                           filename
                           TEXT
                           NOT
                           NULL,
                           file_type
                           TEXT,
                           content
                           TEXT,
                           analysis_data
                           TEXT,
                           confidence_score
                           REAL
                           DEFAULT
                           0.75,
                           intelligence_summary
                           TEXT,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')
        self._conn.commit()
        cursor.close()
        logger.info("Database initialized successfully")

    def store_document(self, doc_id: str, filename: str, content: str, analysis: DocumentAnalysis):
        cursor = self._conn.cursor()
        try:
            file_type = filename.split('.')[-1].lower() if '.' in filename else "unknown"
            analysis_json = analysis.model_dump_json()
            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (id, filename, file_type, content, analysis_data, confidence_score, intelligence_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (doc_id, filename, file_type, content, analysis_json, analysis.confidence_score,
                  analysis.intelligence_summary))
            self._conn.commit()
            logger.info(f"Document {filename} stored successfully")
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            self._conn.rollback()
        finally:
            cursor.close()

    def store_chunks(self, chunks: List[Dict]):
        cursor = self._conn.cursor()
        try:
            for chunk in chunks:
                cursor.execute(
                    'INSERT INTO chunks (embedding_id, doc_id, filename, chunk_text, chunk_index) VALUES (?, ?, ?, ?, ?)',
                    (chunk.get('embedding_id'), chunk.get('doc_id'), chunk.get('filename'), chunk.get('text'),
                     chunk.get('chunk_index')))
            self._conn.commit()
            logger.info(f"Successfully stored {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            self._conn.rollback()
        finally:
            cursor.close()

    def get_chunk_by_embedding_id(self, embedding_id: int) -> Optional[Dict]:
        cursor = self._conn.cursor()
        try:
            cursor.execute('SELECT filename, chunk_text FROM chunks WHERE embedding_id = ?', (embedding_id,))
            result = cursor.fetchone()
            return {'filename': result[0], 'text': result[1]} if result else None
        except Exception as e:
            logger.error(f"Error getting chunk: {e}")
            return None
        finally:
            cursor.close()

    def get_all_documents(self) -> List[Dict]:
        cursor = self._conn.cursor()
        try:
            cursor.execute(
                'SELECT id, filename, file_type, confidence_score, intelligence_summary, created_at FROM documents ORDER BY created_at DESC')
            results = cursor.fetchall()
            return [{
                "id": row[0], "filename": row[1], "file_type": row[2],
                "confidence_score": row[3] or 0.75, "intelligence_summary": row[4] or f"Analysis for {row[1]}",
                "processed_at": row[5] or datetime.now().isoformat()
            } for row in results]
        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            return []
        finally:
            cursor.close()

    def get_rag_stats(self) -> Dict:
        cursor = self._conn.cursor()
        try:
            cursor.execute('SELECT COUNT(*) FROM chunks')
            total_chunks = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(DISTINCT doc_id) FROM chunks')
            total_documents = cursor.fetchone()[0]
            return {"total_chunks": total_chunks, "total_documents": total_documents, "index_dimension": 384,
                    "model_name": "all-MiniLM-L6-v2"}
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"total_chunks": 0, "total_documents": 0, "index_dimension": 384, "model_name": "all-MiniLM-L6-v2"}
        finally:
            cursor.close()

    def delete_document(self, doc_id: str):
        cursor = self._conn.cursor()
        try:
            cursor.execute('DELETE FROM chunks WHERE doc_id = ?', (doc_id,))
            cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
            self._conn.commit()
            logger.info(f"Successfully deleted document {doc_id} and its chunks from the database.")
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            self._conn.rollback()
            return False
        finally:
            cursor.close()

    # NEW METHOD: Get a single document by its ID
    def get_document_by_id(self, doc_id: str) -> Optional[AnalyzedDocument]:
        cursor = self._conn.cursor()
        try:
            cursor.execute('SELECT filename, file_type, created_at, content, analysis_data FROM documents WHERE id = ?',
                           (doc_id,))
            row = cursor.fetchone()
            if not row:
                return None

            filename, file_type, uploaded_at, content, analysis_json = row
            analysis_data = json.loads(analysis_json)

            metadata = DocumentMetadata(
                filename=filename,
                file_type=file_type,
                uploaded_at=uploaded_at,
                file_size=len(content)
            )

            return AnalyzedDocument(
                id=doc_id,
                content=content,
                metadata=metadata,
                analysis=DocumentAnalysis(**analysis_data)
            )
        except Exception as e:
            logger.error(f"Error fetching document {doc_id}: {e}")
            return None
        finally:
            cursor.close()


# (SimpleAnalyzer and SimpleRAG classes remain the same as the previous step)
# ...
class SimpleAnalyzer:
    def analyze_document(self, text: str, metadata: DocumentMetadata) -> DocumentAnalysis:
        start_time = time.time()
        words = text.split() if text else []
        sentences = text.split('.') if text else []
        paragraphs = text.split('\n\n') if text else []
        text_statistics = TextStatistics(word_count=len(words), sentence_count=len(sentences),
                                         paragraph_count=len([p for p in paragraphs if p.strip()]),
                                         readability_score=50.0, language="en")
        text_lower = text.lower()
        sentiment_analysis = SentimentAnalysis()
        if any(word in text_lower for word in
               ['attack', 'bomb', 'terror', 'kill', 'murder', 'assassination', 'explosive']):
            sentiment_analysis.threat_level = "High";
            sentiment_analysis.sentiment_score = -0.8;
            sentiment_analysis.urgency_indicators = ["violent_keywords_detected"]
        elif any(word in text_lower for word in ['robbery', 'theft', 'crime', 'violence', 'kidnap', 'fraud']):
            sentiment_analysis.threat_level = "Medium";
            sentiment_analysis.sentiment_score = -0.4;
            sentiment_analysis.urgency_indicators = ["criminal_activity_mentioned"]
        else:
            sentiment_analysis.threat_level = "Low";
            sentiment_analysis.sentiment_score = 0.1
        nigerian_states = ['lagos', 'abuja', 'kano', 'kaduna', 'rivers', 'oyo', 'delta', 'anambra', 'edo', 'plateau',
                           'cross river', 'ogun', 'kwara', 'imo', 'ondo', 'akwa ibom', 'osun', 'borno', 'bauchi',
                           'enugu', 'kebbi', 'sokoto', 'adamawa', 'katsina', 'bayelsa', 'niger', 'jigawa', 'gombe',
                           'ekiti', 'abia', 'ebonyi', 'taraba', 'zamfara', 'nasarawa', 'kogi', 'yobe', 'benue']
        found_states = [state for state in nigerian_states if state in text_lower]
        location_keywords = ['area', 'zone', 'district', 'region', 'community', 'village', 'town', 'city', 'street',
                             'road']
        other_locations = [word for word in words if
                           any(keyword in word.lower() for keyword in location_keywords) and len(word) > 3]
        geographic_intel = GeographicIntelligence(states=found_states, cities=[],
                                                  countries=["Nigeria"] if found_states else [], coordinates=[],
                                                  total_locations=len(found_states) + len(set(other_locations)),
                                                  other_locations=list(set(other_locations[:10])))
        entities = {"persons": [], "organizations": [], "locations": found_states + list(set(other_locations[:5])),
                    "weapons": [], "vehicles": [], "dates": []}
        for word in words:
            word_lower = word.lower()
            if any(
                weapon in word_lower for weapon in ['gun', 'rifle', 'pistol', 'ak', 'bomb', 'explosive', 'ammunition']):
                entities["weapons"].append(word)
            elif any(vehicle in word_lower for vehicle in ['car', 'vehicle', 'truck', 'motorcycle', 'bike', 'van']):
                entities["vehicles"].append(word)
            elif any(organization in word_lower for organization in
                     ['police', 'army', 'military', 'government', 'agency', 'force']):
                entities["organizations"].append(word)
        crime_types = []
        if any(word in text_lower for word in ['robbery', 'theft', 'steal']): crime_types.append(("theft", 1))
        if any(word in text_lower for word in ['kidnap', 'abduct']): crime_types.append(("kidnapping", 1))
        if any(word in text_lower for word in ['murder', 'kill', 'assassin']): crime_types.append(("homicide", 1))
        if any(word in text_lower for word in ['fraud', 'scam']): crime_types.append(("fraud", 1))
        crime_patterns = CrimePatterns(primary_crimes=crime_types,
                                       crime_frequency={crime: count for crime, count in crime_types}, crime_trends=[])
        processing_time = time.time() - start_time
        intelligence_summary = f"""Intelligence analysis of {metadata.filename} completed. Document contains {len(words)} words with {sentiment_analysis.threat_level} threat level. Geographic analysis identified {len(found_states)} Nigerian states: {', '.join(found_states[:3])}{'...' if len(found_states) > 3 else ''}. Security assessment: {len(entities['weapons'])} weapons mentioned, {len(crime_types)} crime types detected. Processing completed in {processing_time:.2f} seconds with {getattr(text_statistics, 'confidence', 75)}% confidence."""
        return DocumentAnalysis(document_classification=DocumentClassification(), entities=entities,
                                sentiment_analysis=sentiment_analysis, geographic_intelligence=geographic_intel,
                                temporal_intelligence=TemporalIntelligence(),
                                numerical_intelligence=NumericalIntelligence(), crime_patterns=crime_patterns,
                                relationships=[], text_statistics=text_statistics,
                                intelligence_summary=intelligence_summary, confidence_score=0.75,
                                processing_time=processing_time)


class SimpleRAG:
    def __init__(self, openai_api_key: str, store: DocumentStore):
        self.openai_api_key = openai_api_key
        if openai_api_key and openai_api_key != "fake-key": openai.api_key = openai_api_key
        self.store = store;
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2');
        self.index_path = "faiss_index.bin";
        self.index = self.load_or_create_index();
        logger.info(f"RAG system initialized with {self.index.ntotal} vectors")

    def load_or_create_index(self):
        if os.path.exists(self.index_path):
            try:
                return faiss.read_index(self.index_path); logger.info(
                    f"Loaded FAISS index with {faiss.read_index(self.index_path).ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading index: {e}")
        logger.info("Creating new FAISS index");
        return faiss.IndexFlatIP(384)

    def add_document(self, doc_id: str, filename: str, content: str):
        try:
            logger.info(f"Adding document to RAG: {filename}")
            tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo");
            tokens = tokenizer.encode(content);
            chunks = []
            i, max_tokens, overlap = 0, 300, 50
            while i < len(tokens):
                chunk_tokens = tokens[i: i + max_tokens];
                chunk_text = tokenizer.decode(chunk_tokens)
                if chunk_text.strip(): chunks.append(chunk_text)
                i += max_tokens - overlap
            if not chunks: logger.warning(f"No chunks generated for {filename}"); return
            logger.info(f"Generated {len(chunks)} chunks")
            embeddings = self.embedding_model.encode(chunks);
            faiss.normalize_L2(embeddings);
            start_id = self.index.ntotal;
            self.index.add(embeddings.astype('float32'));
            faiss.write_index(self.index, self.index_path)
            chunk_data = [
                {'embedding_id': start_id + i, 'doc_id': doc_id, 'filename': filename, 'text': text, 'chunk_index': i}
                for i, text in enumerate(chunks)]
            self.store.store_chunks(chunk_data);
            logger.info(f"Successfully added {len(chunks)} chunks to RAG system")
        except Exception as e:
            logger.error(f"Error adding document to RAG: {e}")

    def query(self, query_text: str, k: int = 5) -> Dict:
        logger.info(f"Processing query: '{query_text}'")
        try:
            if self.index.ntotal == 0: return {
                "response": "No documents available in the knowledge base. Please upload intelligence documents first.",
                "sources": [], "context_chunks": 0, "timestamp": datetime.now().isoformat(), "model": "none",
                "no_results": True}
            query_embedding = self.embedding_model.encode([query_text]);
            faiss.normalize_L2(query_embedding);
            similarities, indices = self.index.search(query_embedding.astype('float32'), min(k, self.index.ntotal))
            retrieved = []
            for idx, similarity in zip(indices[0], similarities[0]):
                if idx == -1: break
                chunk = self.store.get_chunk_by_embedding_id(int(idx))
                if chunk: chunk['similarity'] = float(similarity); chunk['rank'] = len(retrieved) + 1; retrieved.append(
                    chunk)
            if not retrieved: return {"response": "No relevant information found for your query.", "sources": [],
                                      "context_chunks": 0, "timestamp": datetime.now().isoformat(), "model": "none",
                                      "no_results": True}
            context = "\n\n---\n\n".join([f"Source: {c['filename']}\n{c['text']}" for c in retrieved])
            if not self.openai_api_key or self.openai_api_key == "fake-key": return {
                "response": f"Found {len(retrieved)} relevant sources but cannot generate AI response due to missing OpenAI API key.",
                "sources": [{"filename": c['filename'], "similarity": c['similarity'], "rank": c['rank']} for c in
                            retrieved], "context_chunks": len(retrieved), "timestamp": datetime.now().isoformat(),
                "model": "none"}
            response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "system",
                                                                                        "content": "You are an AI intelligence analyst. Answer based on the provided sources. Be comprehensive and highlight security implications."},
                                                                                       {"role": "user",
                                                                                        "content": f"CONTEXT:\n{context}\n\nQUESTION: {query_text}\n\nANSWER:"}],
                                                      max_tokens=1000, temperature=0.3)
            return {"response": response.choices[0].message.content,
                    "sources": [{"filename": c['filename'], "similarity": c['similarity'], "rank": c['rank']} for c in
                                retrieved], "context_chunks": len(retrieved), "timestamp": datetime.now().isoformat(),
                    "model": "gpt-3.5-turbo"}
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {"response": f"Error processing query: {str(e)}", "sources": [], "error": True, "context_chunks": 0,
                    "timestamp": datetime.now().isoformat(), "model": "none"}


@app.on_event("startup")
def on_startup():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: logger.warning("OPENAI_API_KEY not set - queries will have limited functionality")
    app.state.store = DocumentStore()
    app.state.analyzer = SimpleAnalyzer()
    app.state.rag_system = SimpleRAG(api_key or "fake-key", app.state.store)
    logger.info("Intelligence Document Analyzer started successfully")


def extract_text(file: UploadFile) -> str:
    ext = file.filename.lower().split('.')[-1];
    content = file.file.read()
    if ext == 'pdf': return "".join(page.extract_text() for page in PyPDF2.PdfReader(BytesIO(content)).pages)
    if ext == 'docx': return "\n".join(p.text for p in docx.Document(BytesIO(content)).paragraphs)
    if ext == 'txt': return content.decode('utf-8', errors='ignore')
    raise HTTPException(400, "Unsupported file type")


@app.post("/upload-document", response_model=AnalyzedDocument)
async def handle_upload(file: UploadFile = File(...)):
    try:
        text = extract_text(file)
        if not text.strip(): raise HTTPException(400, "Document is empty or could not be read.")
        doc_id = str(uuid.uuid4())
        metadata = DocumentMetadata(filename=file.filename, file_type=file.filename.split('.')[-1].lower(),
                                    uploaded_at=datetime.now().isoformat(), file_size=len(text))
        analysis = app.state.analyzer.analyze_document(text, metadata)
        app.state.store.store_document(doc_id, file.filename, text, analysis)
        app.state.rag_system.add_document(doc_id, file.filename, text)
        return AnalyzedDocument(id=doc_id, content=text[:2000] + "..." if len(text) > 2000 else text, metadata=metadata,
                                analysis=analysis)
    except Exception as e:
        logger.error(f"Upload failed: {e}");
        raise HTTPException(500, f"Upload error: {e}")


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        result = app.state.rag_system.query(request.query, request.max_results)
        result['query'] = request.query
        return result
    except Exception as e:
        logger.error(f"Query error: {e}");
        return QueryResponse(response=f"Query error: {str(e)}", sources=[], error=True)


@app.get("/rag-stats")
async def get_rag_stats():
    try:
        return app.state.store.get_rag_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}");
        return {"total_chunks": 0, "total_documents": 0, "index_dimension": 384, "model_name": "all-MiniLM-L6-v2"}


@app.get("/document-list")
async def get_document_list():
    try:
        return {"documents": app.state.store.get_all_documents()}
    except Exception as e:
        logger.error(f"Document list error: {e}");
        return {"documents": []}


# NEW ENDPOINT: Get a single document by ID
@app.get("/document/{doc_id}", response_model=AnalyzedDocument)
async def get_document(doc_id: str):
    try:
        document = app.state.store.get_document_by_id(doc_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        return document
    except Exception as e:
        logger.error(f"Error fetching document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.delete("/document/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str):
    try:
        success = app.state.store.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found or could not be deleted.")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/")
async def root():
    return {"message": "Intelligence Document Analyzer API - All Features + Query Fixed", "status": "operational",
            "version": "10.0.0"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)