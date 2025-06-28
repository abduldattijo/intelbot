# main.py - FINAL VERSION WITH WHITESPACE NORMALIZATION

import os
import uuid
import time
import logging
import json
import sqlite3
import re
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO

import pandas as pd
import spacy
from statsmodels.tsa.arima.model import ARIMAResults

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

app = FastAPI(title="Intelligence Document Analyzer API", version="24.1.0 (Final Extraction Fix)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Global NLP Model & File Paths ---
nlp = None
DB_PATH = "documents.db"
MODEL_PATH = "incident_forecaster.pkl"


# --- Pydantic Models (No changes) ---
class DocumentMetadata(BaseModel): filename: str; file_type: str; uploaded_at: str; file_size: int


class DocumentClassification(BaseModel): primary_type: str = "intelligence_report"; sub_types: List[str] = [
    "security_analysis"]; confidence: float = 0.8; security_classification: str = "RESTRICTED"


class SentimentAnalysis(
    BaseModel): overall_sentiment: str = "neutral"; sentiment_score: float = 0.0; threat_level: str = "Low"; urgency_indicators: \
    List[str] = []


class GeographicIntelligence(BaseModel): states: List[str] = []; cities: List[str] = []; countries: List[
    str] = []; coordinates: List[Dict[str, Any]] = []; total_locations: int = 0; other_locations: List[str] = []


class TemporalIntelligence(BaseModel): dates_mentioned: List[str] = []; time_periods: List[str] = []; months_mentioned: \
    List[str] = []; years_mentioned: List[str] = []; temporal_patterns: List[str] = []


class NumericalIntelligence(BaseModel): incidents: List[int] = []; casualties: List[int] = []; weapons: List[
    int] = []; arrests: List[int] = []; monetary_values: List[float] = []


class CrimePatterns(BaseModel): primary_crimes: List[Tuple[str, int]] = []; crime_frequency: Dict[
    str, int] = {}; crime_trends: List[Dict[str, Any]] = []


class TextStatistics(
    BaseModel): word_count: int; sentence_count: int; paragraph_count: int; readability_score: float = 50.0; language: str = "en"


class DocumentAnalysis(BaseModel): document_classification: DocumentClassification; entities: Dict[str, List[
    str]] = {}; sentiment_analysis: SentimentAnalysis; geographic_intelligence: GeographicIntelligence; temporal_intelligence: TemporalIntelligence; numerical_intelligence: NumericalIntelligence; crime_patterns: CrimePatterns; relationships: \
    List[Dict[
        str, Any]] = []; text_statistics: TextStatistics; intelligence_summary: str; confidence_score: float; processing_time: float


class AnalyzedDocument(BaseModel): id: str; content: str; metadata: DocumentMetadata; analysis: DocumentAnalysis


class QueryRequest(BaseModel): query: str; max_results: int = 5


class QueryResponse(BaseModel): response: str; sources: List[Dict]; query: Optional[str] = None; context_chunks: \
    Optional[int] = 0; timestamp: Optional[str] = None; model: Optional[str] = None; error: Optional[
    bool] = False; no_results: Optional[bool] = False


# --- Database and Application Logic ---
class DocumentStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path;
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False);
        self.init_database()

    def init_database(self):
        cursor = self._conn.cursor();
        cursor.execute('''CREATE TABLE IF NOT EXISTS chunks
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
                          )''');
        cursor.execute('''CREATE TABLE IF NOT EXISTS documents
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
                          )''');
        cursor.execute('''CREATE TABLE IF NOT EXISTS incident_time_series
                          (
                              report_date
                              DATE
                              PRIMARY
                              KEY,
                              total_incidents
                              INTEGER
                              NOT
                              NULL
                          )''')
        self._conn.commit();
        cursor.close();
        logger.info("Database initialized successfully (including time_series table).")

    # <<< FINALIZED DYNAMIC EXTRACTION SECTION >>>
    def extract_and_store_incident_data(self, document_text: str):
        """
        FINALIZED: Reliably extracts incident data by intelligently isolating the first paragraph
        and normalizing whitespace for robust keyword matching.
        """
        logger.info("🧠 Starting DYNAMIC first-paragraph extraction...")

        month_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
        }

        def extract_two_numbers_from_paragraph(paragraph: str):
            """Extracts the first two valid incident numbers from a paragraph."""
            number_pattern = r'\((\d{3,4})\)'  # Finds numbers like (632)
            matches = re.findall(number_pattern, paragraph)

            valid_numbers = []
            for digit in matches:
                count = int(digit)
                if 400 <= count <= 1500:  # Valid incident range
                    valid_numbers.append(count)

            if len(valid_numbers) < 2:
                logger.warning(f"Expected at least 2 numbers, found {len(valid_numbers)}")
                return None, None

            num1, num2 = valid_numbers[0], valid_numbers[1]

            # --- FINAL FIX: NORMALIZE WHITESPACE ---
            # This handles hidden newlines (e.g., "as\nagainst") from PDF extraction
            normalized_paragraph = re.sub(r'\s+', ' ', paragraph.lower())

            if 'as against' in normalized_paragraph:
                return num1, num2  # Pattern is "[current] as against [previous]"
            elif 'from' in normalized_paragraph and 'to' in normalized_paragraph:
                return num2, num1  # Pattern is "from [previous] to [current]"
            else:
                logger.warning("Could not determine order from 'as against' or 'from...to'.")
                return None, None

        # --- Main Extraction Logic ---
        extracted_data = []

        header_pattern = r'RETURNS ON ARMED BANDITRY\s*/?\s*ROBBERY\s*FOR (?:THE MONTH OF\s*)?([A-Z]+)[,\s]*(\d{4})'
        sections = re.split(header_pattern, document_text, flags=re.IGNORECASE)

        for i in range(1, len(sections), 3):
            if i + 2 > len(sections): break

            month_str = sections[i].strip().lower()
            year_str = sections[i + 1].strip()
            section_content = sections[i + 2]

            month_num = month_map.get(month_str)
            if not month_num or year_str != '2020': continue

            # Intelligently isolate the first paragraph by finding the start of the second.
            end_of_first_para_match = re.search(r'\s2\.', section_content)

            if end_of_first_para_match:
                first_paragraph = section_content[:end_of_first_para_match.start()].strip()
            else:
                first_paragraph = section_content[:500].strip()  # Fallback
                logger.warning(f"Could not find paragraph delimiter for {month_str}, using fallback slice.")

            logger.info(f"Processing {month_str.title()}: Paragraph length = {len(first_paragraph)}")

            current_count, previous_count = extract_two_numbers_from_paragraph(first_paragraph)

            if current_count:
                report_date = f"2020-{month_num:02d}-01"
                extracted_data.append((report_date, current_count))
                logger.info(
                    f"✅ Extracted for {month_str.title()}: {current_count} incidents (Previous: {previous_count})")
            else:
                logger.error(f"❌ Failed to extract data for {month_str.title()}")

        # Store in database
        if extracted_data:
            cursor = self._conn.cursor()
            try:
                cursor.executemany(
                    "INSERT OR REPLACE INTO incident_time_series (report_date, total_incidents) VALUES (?, ?)",
                    extracted_data
                )
                self._conn.commit()
                logger.info(f"✅ Successfully stored or updated {len(extracted_data)} months of incident data.")
            except Exception as e:
                logger.error(f"Database error during incident data storage: {e}")
                self._conn.rollback()
            finally:
                cursor.close()
        else:
            logger.warning("No dynamic incident data was extracted!")

    def store_document(self, doc_id: str, filename: str, content: str, analysis: DocumentAnalysis):
        cursor = self._conn.cursor()
        try:
            file_type = filename.split('.')[-1].lower() if '.' in filename else "unknown";
            analysis_json = analysis.model_dump_json()
            cursor.execute(
                ''' INSERT OR REPLACE INTO documents (id, filename, file_type, content, analysis_data, confidence_score, intelligence_summary) VALUES (?, ?, ?, ?, ?, ?, ?) ''',
                (doc_id, filename, file_type, content, analysis_json, analysis.confidence_score,
                 analysis.intelligence_summary));
            self._conn.commit();
            logger.info(f"Document {filename} stored successfully")
        except Exception as e:
            logger.error(f"Error storing document: {e}");
            self._conn.rollback()
        finally:
            cursor.close()

    def store_chunks(self, chunks: List[Dict]):
        cursor = self._conn.cursor()
        try:
            for chunk in chunks: cursor.execute(
                'INSERT INTO chunks (embedding_id, doc_id, filename, chunk_text, chunk_index) VALUES (?, ?, ?, ?, ?)',
                (chunk.get('embedding_id'), chunk.get('doc_id'), chunk.get('filename'), chunk.get('text'),
                 chunk.get('chunk_index')))
            self._conn.commit();
            logger.info(f"Successfully stored {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error storing chunks: {e}");
            self._conn.rollback()
        finally:
            cursor.close()

    def get_chunk_by_embedding_id(self, embedding_id: int) -> Optional[Dict]:
        cursor = self._conn.cursor()
        try:
            cursor.execute('SELECT filename, chunk_text FROM chunks WHERE embedding_id = ?', (embedding_id,));
            result = cursor.fetchone()
            return {'filename': result[0], 'text': result[1]} if result else None
        except Exception as e:
            logger.error(f"Error getting chunk: {e}");
            return None
        finally:
            cursor.close()

    def get_all_documents(self) -> List[Dict]:
        cursor = self._conn.cursor()
        try:
            cursor.execute(
                'SELECT id, filename, file_type, confidence_score, intelligence_summary, created_at FROM documents ORDER BY created_at DESC');
            results = cursor.fetchall()
            return [{"id": row[0], "filename": row[1], "file_type": row[2], "confidence_score": row[3] or 0.75,
                     "intelligence_summary": row[4] or f"Analysis for {row[1]}",
                     "processed_at": row[5] or datetime.now().isoformat()} for row in results]
        except Exception as e:
            logger.error(f"Error getting documents: {e}");
            return []
        finally:
            cursor.close()

    def get_rag_stats(self) -> Dict:
        cursor = self._conn.cursor()
        try:
            cursor.execute('SELECT COUNT(*) FROM chunks');
            total_chunks = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(DISTINCT doc_id) FROM chunks');
            total_documents = cursor.fetchone()[0]
            return {"total_chunks": total_chunks, "total_documents": total_documents, "index_dimension": 384,
                    "model_name": "all-MiniLM-L6-v2"}
        except Exception as e:
            logger.error(f"Error getting stats: {e}");
            return {"total_chunks": 0, "total_documents": 0,
                    "index_dimension": 384, "model_name": "all-MiniLM-L6-v2"}
        finally:
            cursor.close()

    def delete_document(self, doc_id: str):
        cursor = self._conn.cursor()
        try:
            cursor.execute('DELETE FROM chunks WHERE doc_id = ?', (doc_id,));
            cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,));
            self._conn.commit();
            logger.info(f"Successfully deleted document {doc_id} and associated chunks.")
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}");
            self._conn.rollback();
            return False
        finally:
            cursor.close()

    def get_document_by_id(self, doc_id: str) -> Optional[AnalyzedDocument]:
        cursor = self._conn.cursor()
        try:
            cursor.execute('SELECT filename, file_type, created_at, content, analysis_data FROM documents WHERE id = ?',
                           (doc_id,));
            row = cursor.fetchone()
            if not row: return None
            filename, file_type, uploaded_at, content, analysis_json = row;
            analysis_data = json.loads(analysis_json)
            metadata = DocumentMetadata(filename=filename, file_type=file_type, uploaded_at=uploaded_at,
                                        file_size=len(content))
            return AnalyzedDocument(id=doc_id, content=content, metadata=metadata,
                                    analysis=DocumentAnalysis(**analysis_data))
        except Exception as e:
            logger.error(f"Error fetching document {doc_id}: {e}");
            return None
        finally:
            cursor.close()


class SimpleAnalyzer:
    def analyze_document(self, text: str, metadata: DocumentMetadata) -> DocumentAnalysis:
        global nlp
        if nlp is None: raise RuntimeError("SpaCy NLP model not loaded. Check startup logs.")
        start_time = time.time()
        doc = nlp(text)
        words = [token.text for token in doc if not token.is_punct and not token.is_space]
        sentences = list(doc.sents)
        text_statistics = TextStatistics(word_count=len(words), sentence_count=len(sentences),
                                         paragraph_count=len(text.split('\n\n')), language=doc.lang_)
        entities = {"persons": set(), "organizations": set(), "locations": set(), "dates": set(), "weapons": set(),
                    "vehicles": set()}
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if len(ent.text.strip().split()) > 1: entities["persons"].add(ent.text.strip())
            elif ent.label_ == "ORG":
                if ent.text.lower() not in ['the service', 'the federal government', 'security agencies']: entities[
                    "organizations"].add(ent.text.strip())
            elif ent.label_ in ["GPE", "LOC"]:
                entities["locations"].add(ent.text.strip())
            elif ent.label_ == "DATE":
                if not any(day in ent.text.lower() for day in ['month', 'period', 'daily', 'the month']):
                    entities["dates"].add(ent.text.strip())
        text_lower = text.lower()
        weapon_keywords = {'ak 47', 'ak47', 'rifles', 'pistols', 'guns', 'bomb', 'explosive', 'ammunition',
                           'pump action', 'cartridges', 'ieds'}
        for keyword in weapon_keywords:
            if keyword in text_lower: entities["weapons"].add(keyword.upper())
        vehicle_keywords = {'car', 'vehicle', 'truck', 'motorcycle', 'bike', 'van'}
        for keyword in vehicle_keywords:
            if keyword in text_lower: entities["vehicles"].add(keyword)
        nigerian_states = {'lagos', 'abuja', 'kano', 'kaduna', 'rivers', 'oyo', 'delta', 'anambra', 'edo', 'plateau',
                           'cross river', 'ogun', 'kwara', 'imo', 'ondo', 'akwa ibom', 'osun', 'borno', 'bauchi',
                           'enugu', 'kebbi', 'sokoto', 'adamawa', 'katsina', 'bayelsa', 'niger', 'jigawa', 'gombe',
                           'ekiti', 'abia', 'ebonyi', 'taraba', 'zamfara', 'nasarawa', 'kogi', 'yobe', 'benue'}
        found_states = {loc for loc in entities["locations"] if loc.lower() in nigerian_states}
        zones = re.findall(r'(North-West|North-Central|South-South|South-East|North-East|South-West)\s+zone', text,
                           re.IGNORECASE)
        other_locations = set(zones)
        geographic_intel = GeographicIntelligence(states=list(found_states), cities=[],
                                                  countries=["Nigeria"] if found_states or zones else [],
                                                  total_locations=len(entities["locations"]),
                                                  other_locations=list(other_locations))
        theft_count = text_lower.count('theft') + text_lower.count('robbery') + text_lower.count('steal')
        kidnap_count = text_lower.count('kidnap') + text_lower.count('abduct')
        homicide_count = text_lower.count('murder') + text_lower.count('kill') + text_lower.count('assassin')
        fraud_count = text_lower.count('fraud') + text_lower.count('scam')
        crime_types = []
        if theft_count > 0: crime_types.append(("Theft/Robbery", theft_count))
        if kidnap_count > 0: crime_types.append(("Kidnapping", kidnap_count))
        if homicide_count > 0: crime_types.append(("Homicide", homicide_count))
        if fraud_count > 0: crime_types.append(("Fraud", fraud_count))
        crime_patterns = CrimePatterns(primary_crimes=crime_types,
                                       crime_frequency={crime: count for crime, count in crime_types})
        sentiment_analysis = SentimentAnalysis()
        if any(word in text_lower for word in
               ['attack', 'bomb', 'terror', 'kill', 'murder', 'assassination', 'explosive']):
            sentiment_analysis.threat_level = "High";
            sentiment_analysis.sentiment_score = -0.8;
        elif any(word in text_lower for word in ['robbery', 'theft', 'crime', 'violence', 'kidnap', 'fraud']):
            sentiment_analysis.threat_level = "Medium";
            sentiment_analysis.sentiment_score = -0.4;
        processing_time = time.time() - start_time
        intelligence_summary = f"NLP analysis of {metadata.filename} complete. Threat level assessed as {sentiment_analysis.threat_level}. Identified {len(entities['persons'])} persons, {len(entities['organizations'])} organizations, and {len(entities['locations'])} locations. Detected {len(crime_patterns.primary_crimes)} primary crime patterns."
        final_entities = {key: sorted(list(value)) for key, value in entities.items()}
        return DocumentAnalysis(document_classification=DocumentClassification(), entities=final_entities,
                                sentiment_analysis=sentiment_analysis, geographic_intelligence=geographic_intel,
                                temporal_intelligence=TemporalIntelligence(),
                                numerical_intelligence=NumericalIntelligence(), crime_patterns=crime_patterns,
                                relationships=[], text_statistics=text_statistics,
                                intelligence_summary=intelligence_summary, confidence_score=0.85,
                                processing_time=processing_time)


class SimpleRAG:
    def __init__(self, openai_api_key: str, store: DocumentStore):
        self.openai_api_key = openai_api_key;
        self.store = store;
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2');
        self.index_path = "faiss_index.bin";
        self.index = self.load_or_create_index();
        logger.info(f"RAG system initialized with {self.index.ntotal} vectors")

    def load_or_create_index(self):
        if os.path.exists(self.index_path):
            try:
                return faiss.read_index(self.index_path)
            except Exception as e:
                logger.error(f"Error loading index: {e}")
        return faiss.IndexFlatIP(384)

    def add_document(self, doc_id: str, filename: str, content: str):
        try:
            tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo");
            tokens = tokenizer.encode(content);
            chunks = []
            i, max_tokens, overlap = 0, 300, 50
            while i < len(tokens):
                chunk_tokens = tokens[i: i + max_tokens];
                chunk_text = tokenizer.decode(chunk_tokens)
                if chunk_text.strip(): chunks.append(chunk_text)
                i += max_tokens - overlap
            if not chunks: return
            embeddings = self.embedding_model.encode(chunks);
            faiss.normalize_L2(embeddings);
            start_id = self.index.ntotal;
            self.index.add(embeddings.astype('float32'));
            faiss.write_index(self.index, self.index_path)
            chunk_data = [
                {'embedding_id': start_id + i, 'doc_id': doc_id, 'filename': filename, 'text': text, 'chunk_index': i}
                for i, text in enumerate(chunks)]
            self.store.store_chunks(chunk_data)
        except Exception as e:
            logger.error(f"Error adding document to RAG: {e}")

    def query(self, query_text: str, k: int = 5) -> Dict:
        try:
            if self.index.ntotal == 0: return {
                "response": "No documents in knowledge base. Please upload a document first.", "sources": [],
                "no_results": True}
            query_embedding = self.embedding_model.encode([query_text]);
            faiss.normalize_L2(query_embedding);
            similarities, indices = self.index.search(query_embedding.astype('float32'), min(k, self.index.ntotal))
            retrieved = [];
            for idx, similarity in zip(indices[0], similarities[0]):
                if idx == -1: break
                chunk = self.store.get_chunk_by_embedding_id(int(idx))
                if chunk:
                    chunk['similarity'] = float(similarity)
                    chunk['rank'] = len(retrieved) + 1
                    retrieved.append(chunk)
            if not retrieved: return {
                "response": "Could not find any relevant information in the documents to answer your query.",
                "sources": [], "no_results": True}
            context = "\n\n---\n\n".join([f"Source: {c['filename']}\n{c['text']}" for c in retrieved])
            client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            response = client.chat.completions.create(model="phi3", messages=[{"role": "system",
                                                                               "content": "You are an AI intelligence analyst. Answer the question based *only* on the provided sources. Cite the source filename for any information you use."},
                                                                              {"role": "user",
                                                                               "content": f"CONTEXT:\n{context}\n\nQUESTION: {query_text}\n\nANSWER:"}],
                                                      max_tokens=1000, temperature=0.1)
            return {"response": response.choices[0].message.content,
                    "sources": [{"filename": c['filename'], "similarity": c['similarity'], "rank": c['rank']} for c in
                                retrieved], "context_chunks": len(retrieved), "timestamp": datetime.now().isoformat(),
                    "model": "phi3"}
        except Exception as e:
            logger.error(f"Query error: {e}");
            return {"response": f"An error occurred while processing the query with the local model: {str(e)}",
                    "sources": [], "error": True}


@app.on_event("startup")
def on_startup():
    global nlp
    api_key = os.getenv("OPENAI_API_KEY", "self-hosted");
    openai.api_key = api_key
    app.state.store = DocumentStore();
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("SpaCy NLP model 'en_core_web_sm' loaded successfully.")
    except OSError:
        logger.error("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
        nlp = None
    app.state.analyzer = SimpleAnalyzer();
    app.state.rag_system = SimpleRAG(api_key, app.state.store);
    logger.info("Intelligence Document Analyzer started with DYNAMIC EXTRACTION mode.")


def extract_text(file: UploadFile) -> str:
    ext = file.filename.lower().split('.')[-1];
    if ext == 'pdf': return "".join(page.extract_text() for page in PyPDF2.PdfReader(file.file).pages)
    if ext == 'docx': return "\n".join(p.text for p in docx.Document(file.file).paragraphs)
    if ext == 'txt': return file.file.read().decode('utf-8', errors='ignore')
    raise HTTPException(400, "Unsupported file type")


# --- API Endpoints ---
@app.post("/upload-document", response_model=AnalyzedDocument)
async def handle_upload(file: UploadFile = File(...)):
    try:
        content_bytes = await file.read()
        await file.seek(0)
        text = extract_text(file)

        if not text.strip(): raise HTTPException(400, "Document is empty")

        app.state.store.extract_and_store_incident_data(text)

        doc_id = str(uuid.uuid4());
        metadata = DocumentMetadata(filename=file.filename, file_type=file.filename.split('.')[-1].lower(),
                                    uploaded_at=datetime.now().isoformat(), file_size=len(content_bytes))
        analysis = app.state.analyzer.analyze_document(text, metadata)
        app.state.store.store_document(doc_id, file.filename, text, analysis);
        app.state.rag_system.add_document(doc_id, file.filename, text)
        return AnalyzedDocument(id=doc_id, content=text[:2000] + "..." if len(text) > 2000 else text, metadata=metadata,
                                analysis=analysis)
    except Exception as e:
        logger.error(f"Upload failed: {e}");
        raise HTTPException(500, f"Upload error: {e}")


@app.get("/forecast")
async def get_forecast_data():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail=f"Forecasting model not found. Please run train_model.py first.")

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT report_date, total_incidents FROM incident_time_series ORDER BY report_date ASC",
                               conn, parse_dates=['report_date'])
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load historical data from database: {e}")

    if df.empty or len(df) < 2:
        raise HTTPException(status_code=404, detail="Not enough historical incident data found to generate a forecast.")

    df.set_index('report_date', inplace=True)
    df = df.asfreq('MS')
    df['total_incidents'] = df['total_incidents'].interpolate()

    with open(MODEL_PATH, 'rb') as pkl_file:
        trained_model: ARIMAResults = pickle.load(pkl_file)

    forecast_steps = 6
    forecast_result = trained_model.get_forecast(steps=forecast_steps)
    forecast_df = forecast_result.summary_frame()

    forecast_data = []
    for date, row in df.iterrows():
        forecast_data.append(
            {"date": date.strftime('%Y-%m-%d'), "incidents": int(row["total_incidents"]), "predicted_incidents": None})

    if forecast_data:
        last_actual_value = forecast_data[-1]["incidents"]
        forecast_data[-1]["predicted_incidents"] = int(last_actual_value)

    for date, row in forecast_df.iterrows():
        forecast_data.append(
            {"date": date.strftime('%Y-%m-%d'), "incidents": None, "predicted_incidents": int(row['mean'])})

    latest_actual = df['total_incidents'].iloc[-1]
    second_latest_actual = df['total_incidents'].iloc[-2]
    predicted_change = ((
                                latest_actual - second_latest_actual) / second_latest_actual) * 100 if second_latest_actual > 0 else 0
    current_threat_level = min(99, (latest_actual / 1000) * 100)

    threat_metrics = {
        "current_threat_level": current_threat_level,
        "predicted_change": round(predicted_change, 1),
        "confidence_score": 85.0,
        "risk_factors": ["High incident volatility month-to-month", "Proliferation of small arms"],
        "recommendations": ["Increase surveillance in high-incident states", "Sustain social intervention programs."]
    }

    return {"forecastData": forecast_data, "threatMetrics": threat_metrics}


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        result = app.state.rag_system.query(request.query, request.max_results);
        result['query'] = request.query
        return result
    except Exception as e:
        logger.error(f"Query error: {e}");
        return QueryResponse(response=f"Query error: {str(e)}", sources=[],
                             error=True)


@app.get("/rag-stats")
async def get_rag_stats():
    try:
        return app.state.store.get_rag_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}");
        return {"total_chunks": 0, "total_documents": 0}


@app.get("/document-list")
async def get_document_list():
    try:
        return {"documents": app.state.store.get_all_documents()}
    except Exception as e:
        logger.error(f"Document list error: {e}");
        return {"documents": []}


@app.get("/document/{doc_id}", response_model=AnalyzedDocument)
async def get_document(doc_id: str):
    try:
        document = app.state.store.get_document_by_id(doc_id)
        if document is None: raise HTTPException(status_code=404, detail="Document not found.")
        return document
    except Exception as e:
        logger.error(f"Error fetching document {doc_id}: {e}");
        raise HTTPException(status_code=500,
                            detail=f"Internal server error: {e}")


@app.delete("/document/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str):
    try:
        success = app.state.store.delete_document(doc_id)
        if not success: raise HTTPException(status_code=404, detail="Document not found.")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}");
        raise HTTPException(status_code=500,
                            detail=f"Internal server error: {e}")


@app.get("/")
async def root():
    return {"message": "Intelligence Document Analyzer API", "status": "operational",
            "version": "24.1.0 (Final Extraction Fix)"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)