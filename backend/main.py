# main.py - FINAL COMPLETE VERSION - All Features & Fixes

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
import io

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

app = FastAPI(title="Intelligence Document Analyzer API", version="34.0.0 (Final Complete)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Global NLP Model & File Paths ---
nlp = None
DB_PATH = "documents.db"
MODEL_PATH = "incident_forecaster.pkl"


# --- Pydantic Models ---
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


class MonthlyComparisonData(BaseModel): metric: str; value1: str; value2: str; change: str


class ComparisonResponse(BaseModel): month1: str; month2: str; comparison_table: List[
    MonthlyComparisonData]; ai_inference: str


# --- Database and Application Logic ---
class DocumentStore:
    def __init__(self, db_path: str = DB_PATH):
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
                           month
                           INTEGER,
                           year
                           INTEGER,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')
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
                              REAL,
                              intelligence_summary
                              TEXT,
                              created_at
                              TIMESTAMP
                              DEFAULT
                              CURRENT_TIMESTAMP
                          )''')
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
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS monthly_detailed_stats
                       (
                           report_date
                           DATE
                           PRIMARY
                           KEY,
                           total_casualties
                           INTEGER,
                           civilian_casualties
                           INTEGER,
                           security_casualties
                           INTEGER,
                           robber_casualties
                           INTEGER,
                           total_arrests
                           INTEGER,
                           robber_arrests
                           INTEGER,
                           murderer_arrests
                           INTEGER,
                           rustler_arrests
                           INTEGER,
                           rapist_arrests
                           INTEGER
                       )
                       ''')
        self._conn.commit()
        cursor.close()
        logger.info("Database initialized successfully.")

    def extract_and_store_incident_data(self, document_text: str):
        logger.info("Extracting main incident data...")
        month_map = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6, 'july': 7, 'august': 8,
                     'september': 9, 'october': 10, 'november': 11, 'december': 12}
        extracted_data = []
        header_pattern = r'RETURNS ON ARMED BANDITRY\s*/?\s*ROBBERY\s*FOR (?:THE MONTH OF\s*)?([A-Z]+)[,\s]*(\d{4})'
        sections = re.split(header_pattern, document_text, flags=re.IGNORECASE)
        for i in range(1, len(sections), 3):
            if i + 2 > len(sections): continue
            month_str, year_str, section_content = sections[i].strip().lower(), sections[i + 1].strip(), sections[i + 2]
            month_num = month_map.get(month_str)
            if not month_num or year_str != '2020': continue
            end_of_first_para_match = re.search(r'\s2\.', section_content)
            first_paragraph = section_content[
                              :end_of_first_para_match.start()].strip() if end_of_first_para_match else section_content[
                                                                                                        :500].strip()
            number_pattern = r'\((\d{3,4})\)'
            matches = [int(d) for d in re.findall(number_pattern, first_paragraph) if 400 <= int(d) <= 1500]
            if len(matches) < 2: continue
            normalized_paragraph = re.sub(r'\s+', ' ', first_paragraph.lower())
            current_count = matches[1] if 'from' in normalized_paragraph and 'to' in normalized_paragraph else matches[
                0]
            if current_count:
                extracted_data.append((f"{year_str}-{month_num:02d}-01", current_count))
        if extracted_data:
            with self._conn:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO incident_time_series (report_date, total_incidents) VALUES (?, ?)",
                    extracted_data)
            logger.info(f"Stored main incident data for {len(set(d[0] for d in extracted_data))} months.")

    def extract_and_store_detailed_stats(self, document_text: str):
        logger.info("Extracting detailed stats...")
        month_map = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6, 'july': 7, 'august': 8,
                     'september': 9, 'october': 10, 'november': 11, 'december': 12}
        header_pattern = r'RETURNS ON ARMED BANDITRY\s*/?\s*ROBBERY\s*FOR (?:THE MONTH OF\s*)?([A-Z]+)[,\s]*(\d{4})'
        sections = re.split(header_pattern, document_text, flags=re.IGNORECASE)
        stats_to_store = []
        for i in range(1, len(sections), 3):
            if i + 2 > len(sections): continue
            month_str, year_str, section_content = sections[i].strip().lower(), sections[i + 1].strip(), sections[i + 2]
            month_num = month_map.get(month_str)
            if not month_num: continue
            report_date = f"{year_str}-{month_num:02d}-01"

            def find_stat(pattern):
                match = re.search(pattern, section_content, re.IGNORECASE)
                return int(match.group(1).replace(',', '')) if match else 0

            total_cas = find_stat(r'\((\d+,?\d*)\)\s*persons,')
            civ_cas = find_stat(r'\((\d+,?\d*)\)\s*civilians')
            rob_cas = find_stat(r'\((\d+,?\d*)\)\s*armed\s+robbers') or find_stat(r'\((\d+,?\d*)\)\s*criminals')
            sec_cas = find_stat(r'\((\d+,?\d*)\)\s*security\s+personnel')

            total_arr = find_stat(r'arrest of about .*?\((\d+,?\d*)\)\s*suspects')
            rob_arr = find_stat(r'\((\d+,?\d*)\)\s*armed\s+robbers')
            mur_arr = find_stat(r'\((\d+)\)\s*murderers')
            rus_arr = find_stat(r'\((\d+)\)\s*cattle\s+rustlers')
            rap_arr = find_stat(r'\((\d+)\)\s*rapists')

            stats_to_store.append(
                (report_date, total_cas, civ_cas, sec_cas, rob_cas, total_arr, rob_arr, mur_arr, rus_arr, rap_arr))

        if stats_to_store:
            with self._conn:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO monthly_detailed_stats (report_date, total_casualties, civilian_casualties, security_casualties, robber_casualties, total_arrests, robber_arrests, murderer_arrests, rustler_arrests, rapist_arrests) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    stats_to_store)
            logger.info(f"Stored detailed stats for {len(stats_to_store)} months.")

    def store_document(self, doc_id: str, filename: str, content: str, analysis: DocumentAnalysis):
        with self._conn:
            self._conn.execute(
                'INSERT OR REPLACE INTO documents (id, filename, file_type, content, analysis_data, confidence_score, intelligence_summary) VALUES (?, ?, ?, ?, ?, ?, ?)',
                (doc_id, filename, filename.split('.')[-1].lower(), content, analysis.model_dump_json(),
                 analysis.confidence_score, analysis.intelligence_summary))

    def store_chunks_with_metadata(self, chunks_with_metadata: List[Dict]):
        with self._conn:
            self._conn.executemany(
                'INSERT INTO chunks (embedding_id, doc_id, filename, chunk_text, chunk_index, month, year) VALUES (:embedding_id, :doc_id, :filename, :text, :chunk_index, :month, :year)',
                chunks_with_metadata)
        logger.info(f"Successfully stored {len(chunks_with_metadata)} chunks with metadata.")

    def get_chunk_by_embedding_id(self, embedding_id: int) -> Optional[Dict]:
        cursor = self._conn.cursor()
        cursor.execute('SELECT filename, chunk_text FROM chunks WHERE embedding_id = ?', (embedding_id,))
        result = cursor.fetchone()
        cursor.close()
        return {'filename': result[0], 'text': result[1]} if result else None

    def get_document_by_id(self, doc_id: str) -> Optional[AnalyzedDocument]:
        cursor = self._conn.cursor()
        cursor.execute('SELECT filename, file_type, created_at, content, analysis_data FROM documents WHERE id = ?',
                       (doc_id,))
        row = cursor.fetchone()
        cursor.close()
        if not row: return None
        filename, file_type, uploaded_at, content, analysis_json = row
        metadata = DocumentMetadata(filename=filename, file_type=file_type, uploaded_at=uploaded_at,
                                    file_size=len(content))
        return AnalyzedDocument(id=doc_id, content=content, metadata=metadata,
                                analysis=DocumentAnalysis(**json.loads(analysis_json)))

    def get_document_by_month_and_year(self, year: int, month: int) -> Optional[AnalyzedDocument]:
        month_name = datetime(year, month, 1).strftime("%B").upper()
        search_patterns = [f'%ROBBERY FOR {month_name}, {year}%', f'%ROBBERY FOR THE MONTH OF {month_name}, {year}%']
        cursor = self._conn.cursor()
        for pattern in search_patterns:
            cursor.execute("SELECT id FROM documents WHERE content LIKE ? ORDER BY created_at DESC LIMIT 1", (pattern,))
            result = cursor.fetchone()
            if result:
                cursor.close()
                return self.get_document_by_id(result[0])
        cursor.close()
        return None

    def get_all_documents(self) -> List[Dict]:
        cursor = self._conn.cursor()
        cursor.execute(
            'SELECT id, filename, file_type, confidence_score, intelligence_summary, created_at FROM documents ORDER BY created_at DESC')
        results = cursor.fetchall()
        cursor.close()
        return [{"id": row[0], "filename": row[1], "file_type": row[2], "confidence_score": row[3] or 0.75,
                 "intelligence_summary": row[4] or f"Analysis for {row[1]}",
                 "processed_at": row[5] or datetime.now().isoformat()} for row in results]

    def delete_document(self, doc_id: str):
        with self._conn:
            cursor = self._conn.cursor()
            cursor.execute('DELETE FROM chunks WHERE doc_id = ?', (doc_id,))
            cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
            return cursor.rowcount > 0

    def get_rag_stats(self) -> Dict:
        cursor = self._conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM chunks')
        total_chunks = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(DISTINCT doc_id) FROM chunks')
        total_documents = cursor.fetchone()[0]
        cursor.close()
        return {"total_chunks": total_chunks, "total_documents": total_documents, "index_dimension": 384,
                "model_name": "all-MiniLM-L6-v2"}


class SimpleAnalyzer:
    def analyze_document(self, text: str, metadata: DocumentMetadata) -> DocumentAnalysis:
        global nlp
        if nlp is None: raise RuntimeError("SpaCy NLP model not loaded.")
        start_time = time.time()
        doc = nlp(text)
        text_lower = text.lower()

        text_statistics = TextStatistics(
            word_count=len([token for token in doc if not token.is_punct and not token.is_space]),
            sentence_count=len(list(doc.sents)), paragraph_count=len(text.split('\n\n')), language=doc.lang_)

        entities = {"persons": set(), "organizations": set()}
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.strip().split()) > 1:
                entities["persons"].add(ent.text.strip())
            elif ent.label_ == "ORG" and ent.text.lower() not in ['the service', 'the federal government',
                                                                  'security agencies']:
                entities["organizations"].add(ent.text.strip())

        nigerian_capitals = {'abia': {'lat': 5.5265, 'lng': 7.4906, 'name': 'Umuahia'},
                             'adamawa': {'lat': 9.2000, 'lng': 12.4833, 'name': 'Yola'},
                             'akwa ibom': {'lat': 5.0515, 'lng': 7.9307, 'name': 'Uyo'},
                             'anambra': {'lat': 6.2120, 'lng': 7.0740, 'name': 'Awka'},
                             'bauchi': {'lat': 10.3158, 'lng': 9.8442, 'name': 'Bauchi'},
                             'bayelsa': {'lat': 4.9267, 'lng': 6.2676, 'name': 'Yenagoa'},
                             'benue': {'lat': 7.7340, 'lng': 8.5120, 'name': 'Makurdi'},
                             'borno': {'lat': 11.8311, 'lng': 13.1510, 'name': 'Maiduguri'},
                             'cross river': {'lat': 4.9516, 'lng': 8.3220, 'name': 'Calabar'},
                             'delta': {'lat': 6.1677, 'lng': 6.7337, 'name': 'Asaba'},
                             'ebonyi': {'lat': 6.3248, 'lng': 8.1142, 'name': 'Abakaliki'},
                             'edo': {'lat': 6.3350, 'lng': 5.6037, 'name': 'Benin City'},
                             'ekiti': {'lat': 7.6667, 'lng': 5.2167, 'name': 'Ado-Ekiti'},
                             'enugu': {'lat': 6.5244, 'lng': 7.5112, 'name': 'Enugu'},
                             'gombe': {'lat': 10.2840, 'lng': 11.1610, 'name': 'Gombe'},
                             'imo': {'lat': 5.4840, 'lng': 7.0351, 'name': 'Owerri'},
                             'jigawa': {'lat': 11.7564, 'lng': 9.3388, 'name': 'Dutse'},
                             'kaduna': {'lat': 10.5105, 'lng': 7.4165, 'name': 'Kaduna'},
                             'kano': {'lat': 12.0022, 'lng': 8.5920, 'name': 'Kano'},
                             'katsina': {'lat': 12.9908, 'lng': 7.6018, 'name': 'Katsina'},
                             'kebbi': {'lat': 12.4537, 'lng': 4.1994, 'name': 'Birnin Kebbi'},
                             'kogi': {'lat': 7.7974, 'lng': 6.7337, 'name': 'Lokoja'},
                             'kwara': {'lat': 8.5000, 'lng': 4.5500, 'name': 'Ilorin'},
                             'lagos': {'lat': 6.5962, 'lng': 3.3431, 'name': 'Ikeja'},
                             'nasarawa': {'lat': 8.4833, 'lng': 8.5167, 'name': 'Lafia'},
                             'niger': {'lat': 9.6134, 'lng': 6.5560, 'name': 'Minna'},
                             'ogun': {'lat': 7.1475, 'lng': 3.3619, 'name': 'Abeokuta'},
                             'ondo': {'lat': 7.2571, 'lng': 5.2058, 'name': 'Akure'},
                             'osun': {'lat': 7.7719, 'lng': 4.5567, 'name': 'Oshogbo'},
                             'oyo': {'lat': 7.3775, 'lng': 3.9470, 'name': 'Ibadan'},
                             'plateau': {'lat': 9.8965, 'lng': 8.8583, 'name': 'Jos'},
                             'rivers': {'lat': 4.8156, 'lng': 7.0498, 'name': 'Port Harcourt'},
                             'sokoto': {'lat': 13.0609, 'lng': 5.2476, 'name': 'Sokoto'},
                             'taraba': {'lat': 8.8833, 'lng': 11.3667, 'name': 'Jalingo'},
                             'yobe': {'lat': 11.7469, 'lng': 11.9609, 'name': 'Damaturu'},
                             'zamfara': {'lat': 12.1667, 'lng': 6.6611, 'name': 'Gusau'},
                             'abuja': {'lat': 9.0765, 'lng': 7.3986, 'name': 'Abuja'}}

        incident_counts_by_state = {}
        table_row_pattern = re.compile(r'(\d+)\.\s*([A-Za-z\s\(\)]+?)\s+(\d+)')
        for match in table_row_pattern.finditer(text):
            state_name = match.group(2).lower().replace("(fct)", "abuja").replace("akwa ibom", "akwalbom").strip()
            if state_name in nigerian_capitals:
                incident_count = int(match.group(3))
                incident_counts_by_state[state_name] = incident_counts_by_state.get(state_name, 0) + incident_count

        geocoded_points = []
        for state_key, capital_info in nigerian_capitals.items():
            if state_key in incident_counts_by_state:
                incident_count = incident_counts_by_state[state_key]
                if incident_count >= 100:
                    threat_level = "high"
                elif incident_count >= 25:
                    threat_level = "medium"
                else:
                    threat_level = "low"
                geocoded_points.append({"location_name": capital_info['name'], "latitude": capital_info['lat'],
                                        "longitude": capital_info['lng'], "threat_level": threat_level,
                                        "confidence": 0.95})

        geographic_intel = GeographicIntelligence(states=sorted(list(incident_counts_by_state.keys())),
                                                  coordinates=geocoded_points, total_locations=len(geocoded_points))
        overall_threat = "High" if any(p['threat_level'] == 'high' for p in geocoded_points) else "Medium"
        sentiment_analysis = SentimentAnalysis(threat_level=overall_threat)
        crime_patterns = CrimePatterns(primary_crimes=sorted(
            [("Armed Robbery", len(re.findall(r'armed robbery', text_lower))),
             ("Murder", len(re.findall(r'murder', text_lower)))], key=lambda x: x[1], reverse=True))
        numerical_intelligence = NumericalIntelligence(incidents=list(incident_counts_by_state.values()))
        intelligence_summary = f"Analysis complete. Overall threat: {overall_threat}. Identified {len(geocoded_points)} key locations with threat levels calculated based on {sum(incident_counts_by_state.values())} total reported incidents."

        return DocumentAnalysis(document_classification=DocumentClassification(),
                                entities={k: sorted(list(v)) for k, v in entities.items()},
                                sentiment_analysis=sentiment_analysis, geographic_intelligence=geographic_intel,
                                temporal_intelligence=TemporalIntelligence(),
                                numerical_intelligence=numerical_intelligence, crime_patterns=crime_patterns,
                                relationships=[], text_statistics=text_statistics,
                                intelligence_summary=intelligence_summary, confidence_score=0.9,
                                processing_time=time.time() - start_time)


class SimpleRAG:
    def __init__(self, openai_api_key: str, store: DocumentStore):
        self.openai_api_key = openai_api_key
        self.store = store
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index_path = "faiss_index.bin"
        self.index = self.load_or_create_index()
        logger.info(f"RAG system initialized with {self.index.ntotal} vectors.")

    def load_or_create_index(self):
        if os.path.exists(self.index_path):
            try:
                return faiss.read_index(self.index_path)
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
        return faiss.IndexFlatIP(384)

    def add_document(self, doc_id: str, filename: str, content: str):
        month_map = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6, 'july': 7, 'august': 8,
                     'september': 9, 'october': 10, 'november': 11, 'december': 12}
        try:
            tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            header_pattern = r'RETURNS ON ARMED BANDITRY\s*/?\s*ROBBERY\s*FOR (?:THE MONTH OF\s*)?([A-Z]+)[,\s]*(\d{4})'
            sections = re.split(header_pattern, content, flags=re.IGNORECASE)
            all_chunks_with_metadata = []
            for i in range(1, len(sections), 3):
                if i + 2 > len(sections): continue
                month_str, year_str, section_content = sections[i].strip().lower(), sections[i + 1].strip(), sections[
                    i + 2]
                month_num = month_map.get(month_str)
                if not month_num: continue
                tokens = tokenizer.encode(section_content)
                j, max_tokens, overlap = 0, 300, 50
                while j < len(tokens):
                    chunk_tokens = tokens[j: j + max_tokens]
                    if chunk_text := tokenizer.decode(chunk_tokens).strip():
                        all_chunks_with_metadata.append(
                            {"text": chunk_text, "month": month_num, "year": int(year_str), "doc_id": doc_id,
                             "filename": filename})
                    j += max_tokens - overlap
            if not all_chunks_with_metadata: return
            embeddings = self.embedding_model.encode([c['text'] for c in all_chunks_with_metadata])
            faiss.normalize_L2(embeddings)
            start_id = self.index.ntotal
            self.index.add(embeddings.astype('float32'));
            faiss.write_index(self.index, self.index_path)
            for i, chunk_meta in enumerate(all_chunks_with_metadata):
                chunk_meta.update({'embedding_id': start_id + i, 'chunk_index': i})
            self.store.store_chunks_with_metadata(all_chunks_with_metadata)
        except Exception as e:
            logger.error(f"Error adding document to RAG: {e}", exc_info=True)

    def query(self, query_text: str, k: int = 5, filters: Optional[Dict] = None,
              keyword_search: Optional[str] = None) -> Dict:
        if self.index.ntotal == 0: return {"response": "No documents in knowledge base.", "sources": [],
                                           "no_results": True, "error": False}

        allowed_ids = None
        if filters or keyword_search:
            logger.info(f"Applying filters: {filters}, Keyword: {keyword_search}")
            query_parts, params = [], []
            if filters:
                if 'month' in filters: query_parts.append("month = ?"); params.append(filters['month'])
                if 'year' in filters: query_parts.append("year = ?"); params.append(filters['year'])
            if keyword_search:
                query_parts.append("chunk_text LIKE ?")
                params.append(f"%{keyword_search}%")

            with self.store._conn:
                cursor = self.store._conn.cursor()
                sql_query = f"SELECT embedding_id FROM chunks WHERE {' AND '.join(query_parts)}"
                results = cursor.execute(sql_query, params).fetchall()
                allowed_ids = {row[0] for row in results}
            if not allowed_ids: return {"response": f"No information found for the specified criteria.", "sources": [],
                                        "no_results": True, "error": False}

        query_embedding = self.embedding_model.encode([query_text]);
        faiss.normalize_L2(query_embedding)
        search_k = self.index.ntotal if allowed_ids else min(k * 5, self.index.ntotal)
        similarities, indices = self.index.search(query_embedding.astype('float32'), k=search_k)

        final_indices, rank = [], 0
        for idx in indices[0]:
            if rank >= k: break
            if idx != -1 and (allowed_ids is None or idx in allowed_ids):
                final_indices.append(idx)
                rank += 1

        retrieved = [chunk for idx in final_indices if
                     idx != -1 and (chunk := self.store.get_chunk_by_embedding_id(int(idx)))]
        if not retrieved: return {"response": "Could not find relevant information.", "sources": [], "no_results": True,
                                  "error": False}

        context = "\n\n---\n\n".join([f"Source: {c['filename']}\n{c['text']}" for c in retrieved])
        client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        response = client.chat.completions.create(model="phi3", messages=[{"role": "system",
                                                                           "content": "You are an AI intelligence analyst. Answer based *only* on the provided sources."},
                                                                          {"role": "user",
                                                                           "content": f"CONTEXT:\n{context}\n\nQUESTION: {query_text}\n\nANSWER:"}],
                                                  max_tokens=1000, temperature=0.1)

        return {"response": response.choices[0].message.content,
                "sources": [{"filename": c['filename']} for c in retrieved], "context_chunks": len(retrieved),
                "timestamp": datetime.now().isoformat(), "model": "phi3", "no_results": False, "error": False}


@app.on_event("startup")
def on_startup():
    global nlp
    app.state.store = DocumentStore()
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.error(
            "SpaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'"); nlp = None
    app.state.analyzer = SimpleAnalyzer()
    app.state.rag_system = SimpleRAG(os.getenv("OPENAI_API_KEY", "self-hosted"), app.state.store)
    logger.info("Intelligence Document Analyzer started.")


def extract_text(filename: str, content_bytes: bytes) -> str:
    ext = filename.lower().split('.')[-1]
    try:
        if ext == 'pdf':
            pdf_file = io.BytesIO(content_bytes)
            return "".join(page.extract_text() for page in PyPDF2.PdfReader(pdf_file).pages)
        if ext == 'docx':
            docx_file = io.BytesIO(content_bytes)
            doc = docx.Document(docx_file)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        full_text.append(cell.text)
            return '\n'.join(full_text)
        if ext == 'txt':
            return content_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"Error processing {ext.upper()} file: {e}")
        raise HTTPException(status_code=400, detail=f"Could not process the {ext.upper()} file.")
    raise HTTPException(400, f"Unsupported file type: {ext}")


@app.post("/upload-document", response_model=AnalyzedDocument)
async def handle_upload(file: UploadFile = File(...)):
    try:
        content_bytes = await file.read()
        text = extract_text(file.filename, content_bytes)
        if not text.strip(): raise HTTPException(400, "Document is empty or could not be read.")
        app.state.store.extract_and_store_incident_data(text)
        app.state.store.extract_and_store_detailed_stats(text)
        doc_id = str(uuid.uuid4())
        metadata = DocumentMetadata(filename=file.filename, file_type=file.filename.split('.')[-1].lower(),
                                    uploaded_at=datetime.now().isoformat(), file_size=len(content_bytes))
        analysis = app.state.analyzer.analyze_document(text, metadata)
        app.state.store.store_document(doc_id, file.filename, text, analysis)
        app.state.rag_system.add_document(doc_id, file.filename, text)
        return AnalyzedDocument(id=doc_id, content=text[:2000] + "..." if len(text) > 2000 else text, metadata=metadata,
                                analysis=analysis)
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=f"An error occurred during upload: {e}")


@app.get("/forecast")
async def get_forecast_data():
    if not os.path.exists(MODEL_PATH): raise HTTPException(status_code=404, detail="Forecasting model not found.")
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT report_date, total_incidents FROM incident_time_series ORDER BY report_date ASC",
                               conn, parse_dates=['report_date'])
    if len(df) < 2: raise HTTPException(status_code=404, detail="Not enough historical data for a forecast.")
    df.set_index('report_date', inplace=True);
    df = df.asfreq('MS').interpolate()
    with open(MODEL_PATH, 'rb') as pkl_file:
        trained_model: ARIMAResults = pickle.load(pkl_file)
    forecast = trained_model.get_forecast(steps=6).summary_frame()
    forecast_data = [
        {"date": date.strftime('%Y-%m-%d'), "incidents": int(row["total_incidents"]), "predicted_incidents": None} for
        date, row in df.iterrows()]
    if forecast_data: forecast_data[-1]["predicted_incidents"] = forecast_data[-1]["incidents"]
    forecast_data.extend(
        [{"date": date.strftime('%Y-%m-%d'), "incidents": None, "predicted_incidents": int(row['mean'])} for date, row
         in forecast.iterrows()])
    latest, second_latest = df['total_incidents'].iloc[-1], df['total_incidents'].iloc[-2]
    return {"forecastData": forecast_data, "threatMetrics": {"current_threat_level": min(99, (latest / 1000) * 100),
                                                             "predicted_change": round(((
                                                                                                    latest - second_latest) / second_latest) * 100 if second_latest > 0 else 0,
                                                                                       1), "confidence_score": 85.0,
                                                             "risk_factors": ["High incident volatility",
                                                                              "Proliferation of small arms"],
                                                             "recommendations": ["Increase surveillance",
                                                                                 "Sustain social intervention programs."]}}


@app.get("/available-months")
async def get_available_months():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT DISTINCT strftime('%Y-%m-01', report_date) as report_date FROM incident_time_series ORDER BY report_date DESC",
            conn)
    return {"available_months": df['report_date'].tolist()}


@app.post("/compare-months", response_model=ComparisonResponse)
async def handle_comparison(month1: str, month2: str):
    try:
        dt1, dt2 = datetime.fromisoformat(month1.replace('Z', '')), datetime.fromisoformat(month2.replace('Z', ''))
        month1_str, month2_str = dt1.strftime('%Y-%m-01'), dt2.strftime('%Y-%m-01')
        with sqlite3.connect(DB_PATH) as conn:
            df1 = pd.read_sql_query(
                f"SELECT total_incidents FROM incident_time_series WHERE report_date = '{month1_str}'", conn)
            df2 = pd.read_sql_query(
                f"SELECT total_incidents FROM incident_time_series WHERE report_date = '{month2_str}'", conn)
        if df1.empty: raise HTTPException(status_code=404, detail=f"Data not found for {dt1.strftime('%B %Y')}")
        if df2.empty: raise HTTPException(status_code=404, detail=f"Data not found for {dt2.strftime('%B %Y')}")
        incidents1, incidents2 = int(df1.iloc[0]['total_incidents']), int(df2.iloc[0]['total_incidents'])
        doc1, doc2 = app.state.store.get_document_by_month_and_year(dt1.year,
                                                                    dt1.month), app.state.store.get_document_by_month_and_year(
            dt2.year, dt2.month)
        summary1, summary2 = (doc1.analysis.intelligence_summary if doc1 else ""), (
            doc2.analysis.intelligence_summary if doc2 else "")
        threat1, threat2 = (doc1.analysis.sentiment_analysis.threat_level if doc1 else "N/A"), (
            doc2.analysis.sentiment_analysis.threat_level if doc2 else "N/A")
        inc_change_raw = incidents1 - incidents2
        inc_change_pct = (inc_change_raw / incidents2 * 100) if incidents2 > 0 else 0
        prompt = f"Analyze change between {dt2.strftime('%B %Y')} ({incidents2} incidents) and {dt1.strftime('%B %Y')} ({incidents1} incidents). Summaries: '{summary2}' and '{summary1}'. What caused the change? Provide an inference and one recommendation."
        ai_result = app.state.rag_system.query(prompt, k=1)
        return ComparisonResponse(month1=dt1.strftime("%B %Y"), month2=dt2.strftime("%B %Y"),
                                  comparison_table=[
                                      MonthlyComparisonData(metric="Total Incidents", value1=str(incidents1),
                                                            value2=str(incidents2),
                                                            change=f"{inc_change_raw:+} ({inc_change_pct:+.1f}%)"),
                                      MonthlyComparisonData(metric="Threat Level", value1=threat1, value2=threat2,
                                                            change="-")],
                                  ai_inference=ai_result.get("response", "Inference not available."))
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Comparison error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document-list")
async def get_document_list():
    try:
        return {"documents": app.state.store.get_all_documents()}
    except Exception as e:
        logger.error(f"Document list error: {e}");
        return {"documents": []}


@app.get("/document/{doc_id}", response_model=AnalyzedDocument)
async def get_document(doc_id: str):
    document = app.state.store.get_document_by_id(doc_id)
    if document is None: raise HTTPException(status_code=404, detail="Document not found.")
    return document


@app.delete("/document/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document_endpoint(doc_id: str):
    success = app.state.store.delete_document(doc_id)
    if not success: raise HTTPException(status_code=404, detail="Document not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/rag-stats")
async def get_rag_stats_endpoint():
    return app.state.store.get_rag_stats()


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    query = request.query.lower()

    month_map = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6, 'july': 7, 'august': 8,
                 'september': 9, 'october': 10, 'november': 11, 'december': 12}
    filters = {}
    for month_name, month_num in month_map.items():
        if month_name in query:
            filters['month'] = f"{month_num:02d}"
            break
    if '2020' in query: filters['year'] = '2020'

    numerical_phrases = ['how many', 'what was the total', 'what is the total', 'total number of', 'sum of', 'count of']
    is_numerical = any(phrase in query for phrase in numerical_phrases)
    is_assessment = any(kw in query for kw in ['assessment', 'summary', 'conclusion', 'assess'])

    if is_numerical:
        logger.info(f"Handling numerical query with filters: {filters}")

        db_column_map = {'casualties': 'total_casualties', 'killed': 'total_casualties',
                         'civilians': 'civilian_casualties', 'security': 'security_casualties',
                         'arrests': 'total_arrests', 'arrested': 'total_arrests'}
        target_column = next((col for kw, col in db_column_map.items() if kw in query), None)

        table_to_query = 'monthly_detailed_stats'
        if 'incident' in query:
            target_column = 'total_incidents'
            table_to_query = 'incident_time_series'

        if target_column:
            try:
                where_clauses, params = [], []
                if 'month' in filters: where_clauses.append("strftime('%m', report_date) = ?"); params.append(
                    filters['month'])
                if 'year' in filters: where_clauses.append("strftime('%Y', report_date) = ?"); params.append(
                    filters['year'])
                where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

                with sqlite3.connect(DB_PATH) as conn:
                    result = conn.execute(f"SELECT SUM({target_column}) FROM {table_to_query} {where_sql}",
                                          params).fetchone()

                total = result[0] if result and result[0] is not None else 0
                return QueryResponse(
                    response=f"Based on the processed documents, the total number of {target_column.replace('_', ' ')} for the specified period is {total}.",
                    sources=[{"filename": "Database Aggregate"}], query=request.query,
                    timestamp=datetime.now().isoformat(), model="SQL Database")
            except Exception as e:
                logger.error(f"SQL query failed: {e}")
                return QueryResponse(response="Could not calculate the answer from the database.", sources=[],
                                     error=True)

    keyword_search = "ASSESSMENT" if is_assessment else None
    logger.info(f"Handling descriptive query. Filters: {filters}, Keyword Search: {keyword_search}")
    try:
        rag_filters = {}
        if 'month' in filters: rag_filters['month'] = int(filters['month'])
        if 'year' in filters: rag_filters['year'] = int(filters['year'])

        result = app.state.rag_system.query(request.query, request.max_results,
                                            filters=rag_filters if rag_filters else None, keyword_search=keyword_search)
        result['query'] = request.query
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        return QueryResponse(response=f"An error occurred during the query: {e}", sources=[], error=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)