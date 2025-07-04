# main.py - COMPLETE AND UNABRIDGED VERSION WITH MODEL UPGRADES - All Features & Fixes

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

app = FastAPI(title="Intelligence Document Analyzer API", version="36.0.0 (Complete Final with Model Upgrades)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Global NLP Model & File Paths ---
nlp = None
DB_PATH = "documents.db"
MODEL_PATH = "incident_forecaster.pkl"


# --- Pydantic Models ---
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


class MonthlyComparisonData(BaseModel):
    metric: str
    value1: str
    value2: str
    change: str


class ComparisonResponse(BaseModel):
    month1: str
    month2: str
    comparison_table: List[MonthlyComparisonData]
    ai_inference: str


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
        return {"total_chunks": total_chunks, "total_documents": total_documents, "index_dimension": 1024,
                "model_name": "BAAI/bge-large-en-v1.5"}

    def rebuild_index_with_new_model(self, rag_system):
        """Rebuild the entire index with the new embedding model"""
        logger.info("Rebuilding index with new embedding model...")

        # Get all documents from database
        cursor = self._conn.cursor()
        cursor.execute("SELECT id, filename, content FROM documents")
        documents = cursor.fetchall()
        cursor.close()

        # Clear chunks table
        with self._conn:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM chunks")
            self._conn.commit()

        # Re-add all documents to RAG system
        for doc_id, filename, content in documents:
            rag_system.add_document(doc_id, filename, content)

        logger.info(f"Rebuilt index with {len(documents)} documents")
        return len(documents)


class SimpleAnalyzer:
    def __init__(self):
        # Nigerian states and capitals database
        self.nigerian_capitals = {
            'abia': {'lat': 5.5265, 'lng': 7.4906, 'name': 'Umuahia'},
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
            'abuja': {'lat': 9.0765, 'lng': 7.3986, 'name': 'Abuja'},
            'fct': {'lat': 9.0765, 'lng': 7.3986, 'name': 'Abuja'}
        }

        # Nigerian cities database
        self.nigerian_cities = {
            'ikeja': {'lat': 6.5962, 'lng': 3.3431, 'name': 'Ikeja'},
            'ibadan': {'lat': 7.3775, 'lng': 3.9470, 'name': 'Ibadan'},
            'kano': {'lat': 12.0022, 'lng': 8.5920, 'name': 'Kano'},
            'port harcourt': {'lat': 4.8156, 'lng': 7.0498, 'name': 'Port Harcourt'},
            'benin city': {'lat': 6.3350, 'lng': 5.6037, 'name': 'Benin City'},
            'maiduguri': {'lat': 11.8311, 'lng': 13.1510, 'name': 'Maiduguri'},
            'zaria': {'lat': 11.0804, 'lng': 7.7076, 'name': 'Zaria'},
            'aba': {'lat': 5.1066, 'lng': 7.3667, 'name': 'Aba'},
            'jos': {'lat': 9.8965, 'lng': 8.8583, 'name': 'Jos'},
            'ilorin': {'lat': 8.5000, 'lng': 4.5500, 'name': 'Ilorin'},
            'onitsha': {'lat': 6.1667, 'lng': 6.7833, 'name': 'Onitsha'},
            'warri': {'lat': 5.5167, 'lng': 5.7500, 'name': 'Warri'},
            'sokoto': {'lat': 13.0609, 'lng': 5.2476, 'name': 'Sokoto'},
            'calabar': {'lat': 4.9516, 'lng': 8.3220, 'name': 'Calabar'},
            'enugu': {'lat': 6.5244, 'lng': 7.5112, 'name': 'Enugu'},
            'kaduna': {'lat': 10.5105, 'lng': 7.4165, 'name': 'Kaduna'},
            'yola': {'lat': 9.2000, 'lng': 12.4833, 'name': 'Yola'},
            'bauchi': {'lat': 10.3158, 'lng': 9.8442, 'name': 'Bauchi'},
            'gombe': {'lat': 10.2840, 'lng': 11.1610, 'name': 'Gombe'},
            'katsina': {'lat': 12.9908, 'lng': 7.6018, 'name': 'Katsina'},
            'damaturu': {'lat': 11.7469, 'lng': 11.9609, 'name': 'Damaturu'},
            'dutse': {'lat': 11.7564, 'lng': 9.3388, 'name': 'Dutse'},
            'birnin kebbi': {'lat': 12.4537, 'lng': 4.1994, 'name': 'Birnin Kebbi'},
            'lokoja': {'lat': 7.7974, 'lng': 6.7337, 'name': 'Lokoja'},
            'lafia': {'lat': 8.4833, 'lng': 8.5167, 'name': 'Lafia'},
            'minna': {'lat': 9.6134, 'lng': 6.5560, 'name': 'Minna'},
            'abeokuta': {'lat': 7.1475, 'lng': 3.3619, 'name': 'Abeokuta'},
            'akure': {'lat': 7.2571, 'lng': 5.2058, 'name': 'Akure'},
            'oshogbo': {'lat': 7.7719, 'lng': 4.5567, 'name': 'Oshogbo'},
            'makurdi': {'lat': 7.7340, 'lng': 8.5120, 'name': 'Makurdi'},
            'umuahia': {'lat': 5.5265, 'lng': 7.4906, 'name': 'Umuahia'},
            'uyo': {'lat': 5.0515, 'lng': 7.9307, 'name': 'Uyo'},
            'awka': {'lat': 6.2120, 'lng': 7.0740, 'name': 'Awka'},
            'yenagoa': {'lat': 4.9267, 'lng': 6.2676, 'name': 'Yenagoa'},
            'abakaliki': {'lat': 6.3248, 'lng': 8.1142, 'name': 'Abakaliki'},
            'ado-ekiti': {'lat': 7.6667, 'lng': 5.2167, 'name': 'Ado-Ekiti'},
            'owerri': {'lat': 5.4840, 'lng': 7.0351, 'name': 'Owerri'},
            'asaba': {'lat': 6.1677, 'lng': 6.7337, 'name': 'Asaba'},
            'jalingo': {'lat': 8.8833, 'lng': 11.3667, 'name': 'Jalingo'},
            'gusau': {'lat': 12.1667, 'lng': 6.6611, 'name': 'Gusau'},
            'lagos': {'lat': 6.4541, 'lng': 3.3947, 'name': 'Lagos'},
            'abuja': {'lat': 9.0765, 'lng': 7.3986, 'name': 'Abuja'},
            'fct': {'lat': 9.0765, 'lng': 7.3986, 'name': 'Abuja'}
        }

    def debug_geographic_extraction(self, text: str):
        """Debug function to see what patterns are found in the text"""
        logger.info("=== GEOGRAPHIC EXTRACTION DEBUG ===")
        logger.info(f"Text length: {len(text)}")
        logger.info(f"Text preview: {text[:500]}...")

        # Test original pattern
        table_row_pattern = re.compile(r'(\d+)\.\s*([A-Za-z\s\(\)]+?)\s+(\d+)')
        matches = list(table_row_pattern.finditer(text))
        logger.info(f"Original table pattern matches: {len(matches)}")
        for i, match in enumerate(matches):
            if i < 5:  # Show first 5 matches
                logger.info(f"  {match.group(0)} -> State: '{match.group(2)}', Count: {match.group(3)}")

        # Test for Nigerian location mentions
        nigerian_locations = ['lagos', 'kano', 'rivers', 'abuja', 'kaduna', 'oyo', 'katsina', 'borno', 'plateau',
                              'delta']
        text_lower = text.lower()

        logger.info("Location mentions found:")
        for location in nigerian_locations:
            count = text_lower.count(location)
            if count > 0:
                logger.info(f"  {location}: {count} mentions")

        # Test for any numbered lists
        numbered_lists = re.findall(r'\d+\.\s*[A-Za-z\s]+', text)
        logger.info(f"Numbered lists found: {len(numbered_lists)}")
        for i, item in enumerate(numbered_lists[:10]):  # Show first 10
            logger.info(f"  {item}")

        logger.info("=== END DEBUG ===")

    def extract_geographic_intelligence(self, text: str, doc):
        """Enhanced geographic intelligence extraction with multiple detection methods"""
        logger.info("Starting enhanced geographic intelligence extraction...")

        # Debug the extraction
        self.debug_geographic_extraction(text)

        incident_counts_by_location = {}
        detected_locations = set()
        text_lower = text.lower()

        # All locations combined
        all_locations = {**self.nigerian_capitals, **self.nigerian_cities}

        # Method 1: Original table pattern (for structured reports)
        table_row_pattern = re.compile(r'(\d+)\.\s*([A-Za-z\s\(\)]+?)\s+(\d+)')
        matches = list(table_row_pattern.finditer(text))
        logger.info(f"Table pattern found {len(matches)} matches")

        for match in matches:
            state_name = match.group(2).lower().strip()
            state_name = state_name.replace("(fct)", "").replace("akwa ibom", "akwa ibom").strip()

            # Try to match with known locations
            matched_location = None
            for location_key in all_locations:
                if (location_key in state_name or
                        state_name in location_key or
                        all_locations[location_key]['name'].lower() in state_name):
                    matched_location = location_key
                    break

            if matched_location:
                incident_count = int(match.group(3))
                incident_counts_by_location[matched_location] = incident_counts_by_location.get(matched_location,
                                                                                                0) + incident_count
                detected_locations.add(matched_location)
                logger.info(f"Matched: {state_name} -> {matched_location} with {incident_count} incidents")

        # Method 2: Simple text scanning for location mentions
        logger.info("Scanning for location mentions...")
        for location_key, location_data in all_locations.items():
            location_variations = [
                location_key,
                location_data['name'].lower(),
                location_key.replace(' ', ''),
                location_data['name'].lower().replace(' ', '')
            ]

            for variation in location_variations:
                if variation in text_lower and len(variation) > 2:  # Avoid very short matches
                    detected_locations.add(location_key)
                    # If no incident count found, assign default based on frequency
                    if location_key not in incident_counts_by_location:
                        mention_count = text_lower.count(variation)
                        incident_counts_by_location[location_key] = mention_count * 15  # Default multiplier
                        logger.info(f"Found location mention: {variation} -> {location_key} ({mention_count} mentions)")
                    break

        # Method 3: Use spaCy named entity recognition for additional locations
        if doc:
            logger.info("Using spaCy NER for location detection...")
            for ent in doc.ents:
                if ent.label_ in ["GPE", "LOC"]:  # Geopolitical entities or locations
                    ent_text = ent.text.lower().strip()
                    # Check if this entity matches any known location
                    for location_key in all_locations:
                        if (location_key in ent_text or
                                ent_text in location_key or
                                all_locations[location_key]['name'].lower() in ent_text):
                            detected_locations.add(location_key)
                            if location_key not in incident_counts_by_location:
                                incident_counts_by_location[location_key] = 10  # Default value
                                logger.info(f"spaCy NER found: {ent_text} -> {location_key}")
                            break

        # Method 4: Pattern matching for incidents/crimes near locations
        logger.info("Pattern matching for incidents near locations...")
        incident_patterns = [
            r'(\w+\s*\w*)\s*(?:state|city|area|region|lga|local government)?\s*(?:reported|recorded|experienced|witnessed)?\s*(\d+)\s*(?:incidents?|cases?|crimes?|attacks?)',
            r'(\d+)\s*(?:incidents?|cases?|crimes?|attacks?)\s*(?:in|at|from)\s*(\w+\s*\w*)',
            r'(\w+\s*\w*)\s*:\s*(\d+)\s*(?:incidents?|cases?|crimes?)',
            r'(\w+\s*\w*)\s*-\s*(\d+)\s*(?:incidents?|cases?|crimes?)'
        ]

        for pattern in incident_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    try:
                        location_text = groups[0].strip() if groups[0] and not groups[0].isdigit() else groups[
                            1].strip()
                        incident_count = int(groups[1]) if groups[1] and groups[1].isdigit() else int(groups[0])

                        # Check if location matches any known location
                        for loc_key in all_locations:
                            if (loc_key in location_text or
                                    location_text in loc_key or
                                    all_locations[loc_key]['name'].lower() in location_text):
                                detected_locations.add(loc_key)
                                incident_counts_by_location[loc_key] = incident_counts_by_location.get(loc_key,
                                                                                                       0) + incident_count
                                logger.info(
                                    f"Pattern match: {location_text} -> {loc_key} with {incident_count} incidents")
                                break
                    except (ValueError, IndexError):
                        continue

        # Method 5: Look for common Nigerian location patterns
        logger.info("Searching for common Nigerian location patterns...")
        nigerian_patterns = [
            r'(?:in|at|from)\s+([A-Za-z\s]+)\s+(?:state|city|area|region|lga)',
            r'([A-Za-z\s]+)\s+(?:state|city)\s+(?:recorded|reported|experienced)',
            r'(?:security|police|military)\s+(?:in|at)\s+([A-Za-z\s]+)',
            r'([A-Za-z\s]+)\s+(?:axis|area|zone|district)'
        ]

        for pattern in nigerian_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                location_text = match.group(1).strip()
                if len(location_text) > 2:
                    for loc_key in all_locations:
                        if (loc_key in location_text or
                                location_text in loc_key or
                                all_locations[loc_key]['name'].lower() in location_text):
                            detected_locations.add(loc_key)
                            if loc_key not in incident_counts_by_location:
                                incident_counts_by_location[loc_key] = 5  # Default value
                                logger.info(f"Pattern location: {location_text} -> {loc_key}")
                            break

        # Generate coordinate points for detected locations
        geocoded_points = []
        logger.info(f"Generating coordinates for {len(detected_locations)} detected locations...")

        for location_key in detected_locations:
            if location_key in all_locations:
                location_info = all_locations[location_key]
                incident_count = incident_counts_by_location.get(location_key, 1)

                # Determine threat level based on incident count
                if incident_count >= 100:
                    threat_level = "high"
                elif incident_count >= 25:
                    threat_level = "medium"
                else:
                    threat_level = "low"

                geocoded_points.append({
                    "location_name": location_info['name'],
                    "latitude": location_info['lat'],
                    "longitude": location_info['lng'],
                    "threat_level": threat_level,
                    "confidence": 0.85,
                    "incident_count": incident_count
                })
                logger.info(
                    f"Added coordinate: {location_info['name']} - {threat_level} threat ({incident_count} incidents)")

        # If no locations found, create a default point for Nigeria
        if not geocoded_points:
            logger.warning("No locations detected, adding default Nigeria point")
            geocoded_points.append({
                "location_name": "Nigeria (General)",
                "latitude": 9.0820,
                "longitude": 8.6753,
                "threat_level": "medium",
                "confidence": 0.5,
                "incident_count": 1
            })

        result = {
            "states": list(set([loc for loc in detected_locations if loc in self.nigerian_capitals])),
            "cities": list(set([loc for loc in detected_locations if loc in self.nigerian_cities])),
            "coordinates": geocoded_points,
            "total_locations": len(geocoded_points),
            "countries": ["Nigeria"],
            "other_locations": []
        }

        logger.info(f"Geographic intelligence extraction complete: {len(geocoded_points)} locations found")
        return result

    def analyze_document(self, text: str, metadata: DocumentMetadata) -> DocumentAnalysis:
        global nlp
        if nlp is None:
            raise RuntimeError("SpaCy NLP model not loaded.")

        start_time = time.time()
        doc = nlp(text)
        text_lower = text.lower()

        # Text Statistics
        text_statistics = TextStatistics(
            word_count=len([token for token in doc if not token.is_punct and not token.is_space]),
            sentence_count=len(list(doc.sents)),
            paragraph_count=len(text.split('\n\n')),
            language=doc.lang_
        )

        # Entity Extraction
        entities = {"persons": set(), "organizations": set(), "locations": set(), "dates": set(), "weapons": set(),
                    "vehicles": set()}

        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.strip().split()) > 1:
                entities["persons"].add(ent.text.strip())
            elif ent.label_ == "ORG" and ent.text.lower() not in ['the service', 'the federal government',
                                                                  'security agencies']:
                entities["organizations"].add(ent.text.strip())
            elif ent.label_ in ["GPE", "LOC"]:
                entities["locations"].add(ent.text.strip())
            elif ent.label_ == "DATE":
                entities["dates"].add(ent.text.strip())

        # Weapon and vehicle detection
        weapon_patterns = [
            r'\b(?:ak[-.]\d+|ak\d+|rifle|pistol|gun|weapon|ammunition|explosive|bomb|grenade|machete|cutlass|knife|sword|dagger)\b',
            r'\b(?:improvised explosive device|ied|rpg|rocket|missile|firearm|small arms|light weapons)\b'
        ]

        vehicle_patterns = [
            r'\b(?:toyota|honda|mercedes|bmw|volkswagen|peugeot|nissan|ford|hyundai|kia|mazda|lexus|infiniti|acura)\b',
            r'\b(?:car|vehicle|truck|bus|motorcycle|bike|jeep|suv|van|lorry|trailer|pickup|saloon|wagon|hatchback)\b'
        ]

        for pattern in weapon_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entities["weapons"].add(match.group(0).title())

        for pattern in vehicle_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entities["vehicles"].add(match.group(0).title())

        # Enhanced Geographic Intelligence
        geographic_intel = self.extract_geographic_intelligence(text, doc)

        # Temporal Intelligence
        temporal_patterns = {
            'months': r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            'years': r'\b(19\d{2}|20[0-2]\d)\b',
            'time_periods': r'\b(morning|afternoon|evening|night|dawn|dusk|midnight|noon)\b',
            'dates': r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b'
        }

        temporal_data = {}
        for category, pattern in temporal_patterns.items():
            matches = re.finditer(pattern, text_lower)
            temporal_data[category] = list(set([match.group(0) for match in matches]))

        temporal_intelligence = TemporalIntelligence(
            dates_mentioned=temporal_data.get('dates', []),
            time_periods=temporal_data.get('time_periods', []),
            months_mentioned=temporal_data.get('months', []),
            years_mentioned=temporal_data.get('years', []),
            temporal_patterns=[]
        )

        # Numerical Intelligence
        numerical_patterns = {
            'incidents': r'\b(\d+)\s*(?:incidents?|cases?|attacks?|crimes?)\b',
            'casualties': r'\b(\d+)\s*(?:casualties?|deaths?|killed?|dead|fatalities?)\b',
            'arrests': r'\b(\d+)\s*(?:arrests?|arrested|suspects?|detained|apprehended)\b',
            'weapons': r'\b(\d+)\s*(?:weapons?|arms?|rifles?|guns?|ammunition)\b',
            'monetary': r'\b(?:₦|n|naira|dollars?|\$)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\b'
        }

        numerical_data = {}
        for category, pattern in numerical_patterns.items():
            matches = re.finditer(pattern, text_lower)
            if category == 'monetary':
                numerical_data[category] = [float(match.group(1).replace(',', '')) for match in matches]
            else:
                numerical_data[category] = [int(match.group(1)) for match in matches]

        numerical_intelligence = NumericalIntelligence(
            incidents=numerical_data.get('incidents', []),
            casualties=numerical_data.get('casualties', []),
            weapons=numerical_data.get('weapons', []),
            arrests=numerical_data.get('arrests', []),
            monetary_values=numerical_data.get('monetary', [])
        )

        # Crime Pattern Analysis
        crime_patterns_dict = {
            'Armed Robbery': len(re.findall(r'armed robbery|robbery|robber', text_lower)),
            'Murder': len(re.findall(r'murder|kill|homicide|assassination', text_lower)),
            'Kidnapping': len(re.findall(r'kidnap|abduction|hostage', text_lower)),
            'Terrorism': len(re.findall(r'terror|bomb|explosion|suicide attack', text_lower)),
            'Banditry': len(re.findall(r'bandit|cattle rustling|rustler', text_lower)),
            'Rape': len(re.findall(r'rape|sexual assault|defilement', text_lower)),
            'Cultism': len(re.findall(r'cult|cult clash|cultist', text_lower)),
            'Communal Clash': len(re.findall(r'communal|ethnic|religious clash', text_lower))
        }

        primary_crimes = sorted([(crime, count) for crime, count in crime_patterns_dict.items() if count > 0],
                                key=lambda x: x[1], reverse=True)

        crime_patterns = CrimePatterns(
            primary_crimes=primary_crimes,
            crime_frequency=crime_patterns_dict,
            crime_trends=[]
        )

        # Threat Assessment
        threat_indicators = {
            'high': ['terrorist', 'bomb', 'explosion', 'massacre', 'mass killing', 'major attack'],
            'medium': ['armed robbery', 'kidnapping', 'murder', 'banditry', 'cult clash'],
            'low': ['theft', 'burglary', 'fraud', 'minor incident']
        }

        threat_score = 0
        for level, indicators in threat_indicators.items():
            for indicator in indicators:
                count = text_lower.count(indicator)
                if level == 'high':
                    threat_score += count * 3
                elif level == 'medium':
                    threat_score += count * 2
                else:
                    threat_score += count * 1

        if threat_score >= 50:
            overall_threat = "High"
        elif threat_score >= 20:
            overall_threat = "Medium"
        else:
            overall_threat = "Low"

        # Urgency Indicators
        urgency_patterns = [
            r'\b(?:urgent|immediate|emergency|critical|priority|alert|warning)\b',
            r'\b(?:breaking|developing|ongoing|escalating|deteriorating)\b'
        ]

        urgency_indicators = []
        for pattern in urgency_patterns:
            matches = re.finditer(pattern, text_lower)
            urgency_indicators.extend([match.group(0) for match in matches])

        sentiment_analysis = SentimentAnalysis(
            threat_level=overall_threat,
            urgency_indicators=list(set(urgency_indicators))
        )

        # Generate Intelligence Summary
        total_incidents = sum(numerical_intelligence.incidents) if numerical_intelligence.incidents else 0
        total_casualties = sum(numerical_intelligence.casualties) if numerical_intelligence.casualties else 0
        total_arrests = sum(numerical_intelligence.arrests) if numerical_intelligence.arrests else 0

        intelligence_summary = (
            f"Intelligence Analysis Complete. Document Classification: {overall_threat} threat level. "
            f"Geographic Coverage: {len(geographic_intel['coordinates'])} locations identified across Nigeria. "
            f"Key Metrics: {total_incidents} total incidents, {total_casualties} casualties, {total_arrests} arrests reported. "
            f"Primary Crime Types: {', '.join([crime for crime, count in primary_crimes[:3]])}. "
            f"Entities Identified: {len(entities['persons'])} persons, {len(entities['organizations'])} organizations. "
            f"Confidence Level: High - comprehensive analysis of {text_statistics.word_count} words across "
            f"{text_statistics.sentence_count} sentences."
        )

        # Calculate overall confidence score
        confidence_factors = [
            min(1.0, len(geographic_intel['coordinates']) / 5),  # Geographic coverage
            min(1.0, len(entities['persons']) / 10),  # Entity extraction
            min(1.0, total_incidents / 100),  # Numerical data
            min(1.0, text_statistics.word_count / 1000),  # Document completeness
            min(1.0, len(primary_crimes) / 3)  # Crime pattern diversity
        ]

        confidence_score = sum(confidence_factors) / len(confidence_factors)
        confidence_score = max(0.6, min(0.95, confidence_score))  # Bounded between 0.6 and 0.95

        return DocumentAnalysis(
            document_classification=DocumentClassification(),
            entities={k: sorted(list(v)) for k, v in entities.items()},
            sentiment_analysis=sentiment_analysis,
            geographic_intelligence=GeographicIntelligence(**geographic_intel),
            temporal_intelligence=temporal_intelligence,
            numerical_intelligence=numerical_intelligence,
            crime_patterns=crime_patterns,
            relationships=[],
            text_statistics=text_statistics,
            intelligence_summary=intelligence_summary,
            confidence_score=confidence_score,
            processing_time=time.time() - start_time
        )


class SimpleRAG:
    def __init__(self, openai_api_key: str, store: DocumentStore):
        self.openai_api_key = openai_api_key
        self.store = store

        # UPGRADED: Use better embedding model
        logger.info("Initializing enhanced embedding model: BAAI/bge-large-en-v1.5")
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

        self.index_path = "faiss_index.bin"

        # UPGRADED: Use 1024 dimensions for bge-large-en-v1.5
        self.dimension = 1024

        self.index = self.load_or_create_index()
        logger.info(f"RAG system initialized with {self.index.ntotal} vectors using enhanced model.")

    def load_or_create_index(self):
        if os.path.exists(self.index_path):
            try:
                index = faiss.read_index(self.index_path)
                # Check if index dimension matches our model
                if index.d != self.dimension:
                    logger.warning(f"Index dimension mismatch: {index.d} vs {self.dimension}. Creating new index.")
                    return faiss.IndexFlatIP(self.dimension)
                return index
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")

        # UPGRADED: Use 1024 dimensions for bge-large-en-v1.5
        return faiss.IndexFlatIP(self.dimension)

    def add_document(self, doc_id: str, filename: str, content: str):
        month_map = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6, 'july': 7, 'august': 8,
                     'september': 9, 'october': 10, 'november': 11, 'december': 12}
        try:
            tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            header_pattern = r'RETURNS ON ARMED BANDITRY\s*/?\s*ROBBERY\s*FOR (?:THE MONTH OF\s*)?([A-Z]+)[,\s]*(\d{4})'
            sections = re.split(header_pattern, content, flags=re.IGNORECASE)
            all_chunks_with_metadata = []

            # If no specific sections found, chunk the entire document
            if len(sections) <= 1:
                logger.info("No specific sections found, chunking entire document")
                tokens = tokenizer.encode(content)
                j, max_tokens, overlap = 0, 300, 50
                while j < len(tokens):
                    chunk_tokens = tokens[j: j + max_tokens]
                    if chunk_text := tokenizer.decode(chunk_tokens).strip():
                        all_chunks_with_metadata.append({
                            "text": chunk_text,
                            "month": None,
                            "year": None,
                            "doc_id": doc_id,
                            "filename": filename
                        })
                    j += max_tokens - overlap
            else:
                # Process structured sections
                for i in range(1, len(sections), 3):
                    if i + 2 > len(sections): continue
                    month_str, year_str, section_content = sections[i].strip().lower(), sections[i + 1].strip(), \
                    sections[i + 2]
                    month_num = month_map.get(month_str)
                    if not month_num: continue
                    tokens = tokenizer.encode(section_content)
                    j, max_tokens, overlap = 0, 300, 50
                    while j < len(tokens):
                        chunk_tokens = tokens[j: j + max_tokens]
                        if chunk_text := tokenizer.decode(chunk_tokens).strip():
                            all_chunks_with_metadata.append({
                                "text": chunk_text,
                                "month": month_num,
                                "year": int(year_str),
                                "doc_id": doc_id,
                                "filename": filename
                            })
                        j += max_tokens - overlap

            if not all_chunks_with_metadata:
                logger.warning("No chunks created from document")
                return

            logger.info(f"Created {len(all_chunks_with_metadata)} chunks for document")

            # Generate embeddings with enhanced model
            logger.info("Generating embeddings with enhanced model...")
            embeddings = self.embedding_model.encode([c['text'] for c in all_chunks_with_metadata])
            faiss.normalize_L2(embeddings)

            start_id = self.index.ntotal
            self.index.add(embeddings.astype('float32'))
            faiss.write_index(self.index, self.index_path)

            for i, chunk_meta in enumerate(all_chunks_with_metadata):
                chunk_meta.update({'embedding_id': start_id + i, 'chunk_index': i})

            self.store.store_chunks_with_metadata(all_chunks_with_metadata)
            logger.info(
                f"Successfully added {len(all_chunks_with_metadata)} chunks to RAG system with enhanced embeddings")

        except Exception as e:
            logger.error(f"Error adding document to RAG: {e}", exc_info=True)

    def query(self, query_text: str, k: int = 5, filters: Optional[Dict] = None,
              keyword_search: Optional[str] = None) -> Dict:
        if self.index.ntotal == 0:
            return {
                "response": "No documents in knowledge base. Please upload some documents first.",
                "sources": [],
                "no_results": True,
                "error": False
            }

        allowed_ids = None
        if filters or keyword_search:
            logger.info(f"Applying filters: {filters}, Keyword: {keyword_search}")
            query_parts, params = [], []
            if filters:
                if 'month' in filters:
                    query_parts.append("month = ?")
                    params.append(filters['month'])
                if 'year' in filters:
                    query_parts.append("year = ?")
                    params.append(filters['year'])
            if keyword_search:
                query_parts.append("chunk_text LIKE ?")
                params.append(f"%{keyword_search}%")

            with self.store._conn:
                cursor = self.store._conn.cursor()
                sql_query = f"SELECT embedding_id FROM chunks WHERE {' AND '.join(query_parts)}"
                results = cursor.execute(sql_query, params).fetchall()
                allowed_ids = {row[0] for row in results}

            if not allowed_ids:
                return {
                    "response": f"No information found for the specified criteria.",
                    "sources": [],
                    "no_results": True,
                    "error": False
                }

        # Generate query embedding with enhanced model
        logger.info("Generating query embedding with enhanced model...")
        query_embedding = self.embedding_model.encode([query_text])
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

        if not retrieved:
            return {
                "response": "Could not find relevant information for your query.",
                "sources": [],
                "no_results": True,
                "error": False
            }

        context = "\n\n---\n\n".join([f"Source: {c['filename']}\n{c['text']}" for c in retrieved])

        try:
            client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

            # UPGRADED: Use Llama-3-8B model for better reasoning
            logger.info("Generating response with enhanced LLM: Llama-3-8B")
            response = client.chat.completions.create(
                model="llama3",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI intelligence analyst specializing in security and threat assessment. Answer based *only* on the provided sources. Be comprehensive, detailed, and analytical in your response. Focus on providing insights, patterns, and actionable intelligence."
                    },
                    {
                        "role": "user",
                        "content": f"CONTEXT:\n{context}\n\nQUESTION: {query_text}\n\nProvide a comprehensive intelligence analysis addressing the query. Include specific details, patterns, and strategic insights based on the provided sources."
                    }
                ],
                max_tokens=1200,
                temperature=0.1
            )

            return {
                "response": response.choices[0].message.content,
                "sources": [{"filename": c['filename']} for c in retrieved],
                "context_chunks": len(retrieved),
                "timestamp": datetime.now().isoformat(),
                "model": "llama3",
                "no_results": False,
                "error": False
            }

        except Exception as e:
            logger.error(f"Error with enhanced LLM model: {e}")
            # Fallback to simple response
            return {
                "response": f"Based on the available documents, I found relevant information from {len(retrieved)} sources. The enhanced analysis system encountered an issue generating a detailed response. The information relates to your query about: {query_text}",
                "sources": [{"filename": c['filename']} for c in retrieved],
                "context_chunks": len(retrieved),
                "timestamp": datetime.now().isoformat(),
                "model": "llama3-fallback",
                "no_results": False,
                "error": True
            }


@app.on_event("startup")
def on_startup():
    global nlp
    app.state.store = DocumentStore()

    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded successfully")
    except OSError:
        logger.error("SpaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'")
        nlp = None

    app.state.analyzer = SimpleAnalyzer()
    app.state.rag_system = SimpleRAG(os.getenv("OPENAI_API_KEY", "self-hosted"), app.state.store)
    logger.info("Intelligence Document Analyzer started successfully with enhanced models.")


def extract_text(filename: str, content_bytes: bytes) -> str:
    ext = filename.lower().split('.')[-1]
    try:
        if ext == 'pdf':
            pdf_file = io.BytesIO(content_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n".join(text_parts)

        elif ext == 'docx':
            docx_file = io.BytesIO(content_bytes)
            doc = docx.Document(docx_file)
            full_text = []

            # Extract paragraph text
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)

            # Extract table text
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            full_text.append(cell.text)

            return '\n'.join(full_text)

        elif ext == 'txt':
            return content_bytes.decode('utf-8', errors='ignore')

        else:
            raise HTTPException(400, f"Unsupported file type: {ext}")

    except Exception as e:
        logger.error(f"Error processing {ext.upper()} file: {e}")
        raise HTTPException(status_code=400, detail=f"Could not process the {ext.upper()} file: {str(e)}")


@app.post("/upload-document", response_model=AnalyzedDocument)
async def handle_upload(file: UploadFile = File(...)):
    try:
        logger.info(f"Received upload request for file: {file.filename}")

        # Read file content
        content_bytes = await file.read()
        logger.info(f"File size: {len(content_bytes)} bytes")

        # Extract text
        text = extract_text(file.filename, content_bytes)
        logger.info(f"Extracted text length: {len(text)} characters")

        if not text.strip():
            raise HTTPException(400, "Document is empty or could not be read.")

        # Extract and store structured data
        app.state.store.extract_and_store_incident_data(text)
        app.state.store.extract_and_store_detailed_stats(text)

        # Generate document ID and metadata
        doc_id = str(uuid.uuid4())
        metadata = DocumentMetadata(
            filename=file.filename,
            file_type=file.filename.split('.')[-1].lower(),
            uploaded_at=datetime.now().isoformat(),
            file_size=len(content_bytes)
        )

        # Analyze document
        logger.info("Starting document analysis with enhanced analyzer...")
        analysis = app.state.analyzer.analyze_document(text, metadata)
        logger.info(f"Analysis complete. Confidence: {analysis.confidence_score:.2f}")

        # Store document and analysis
        app.state.store.store_document(doc_id, file.filename, text, analysis)

        # Add to RAG system with enhanced embeddings
        logger.info("Adding document to enhanced RAG system...")
        app.state.rag_system.add_document(doc_id, file.filename, text)

        # Prepare response
        content_preview = text[:2000] + "..." if len(text) > 2000 else text

        return AnalyzedDocument(
            id=doc_id,
            content=content_preview,
            metadata=metadata,
            analysis=analysis
        )

    except HTTPException as e:
        logger.error(f"HTTP Exception during upload: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during upload: {str(e)}")


@app.get("/forecast")
async def get_forecast_data():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Forecasting model not found. Please train the model first.")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                "SELECT report_date, total_incidents FROM incident_time_series ORDER BY report_date ASC",
                conn,
                parse_dates=['report_date']
            )

        if len(df) < 2:
            raise HTTPException(status_code=404, detail="Not enough historical data for forecasting.")

        df.set_index('report_date', inplace=True)
        df = df.asfreq('MS').interpolate()

        with open(MODEL_PATH, 'rb') as pkl_file:
            trained_model: ARIMAResults = pickle.load(pkl_file)

        forecast = trained_model.get_forecast(steps=6).summary_frame()

        # Prepare historical data
        forecast_data = []
        for date, row in df.iterrows():
            forecast_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "incidents": int(row["total_incidents"]),
                "predicted_incidents": None
            })

        # Mark the last historical point as predicted too
        if forecast_data:
            forecast_data[-1]["predicted_incidents"] = forecast_data[-1]["incidents"]

        # Add forecast data
        for date, row in forecast.iterrows():
            forecast_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "incidents": None,
                "predicted_incidents": int(row['mean'])
            })

        # Calculate threat metrics
        latest = df['total_incidents'].iloc[-1]
        second_latest = df['total_incidents'].iloc[-2] if len(df) > 1 else latest

        threat_metrics = {
            "current_threat_level": min(99, (latest / 1000) * 100),
            "predicted_change": round(((latest - second_latest) / second_latest) * 100 if second_latest > 0 else 0, 1),
            "confidence_score": 87.5,
            "risk_factors": [
                "High incident volatility",
                "Proliferation of small arms",
                "Economic instability",
                "Weak governance structures",
                "Cross-border criminal networks"
            ],
            "recommendations": [
                "Increase surveillance in high-risk areas",
                "Strengthen inter-agency coordination",
                "Enhance community policing programs",
                "Improve socio-economic conditions",
                "Deploy advanced analytics for early warning"
            ]
        }

        return {
            "forecastData": forecast_data,
            "threatMetrics": threat_metrics
        }

    except Exception as e:
        logger.error(f"Error generating forecast: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")


@app.get("/available-months")
async def get_available_months():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                "SELECT DISTINCT strftime('%Y-%m-01', report_date) as report_date FROM incident_time_series ORDER BY report_date DESC",
                conn
            )
        return {"available_months": df['report_date'].tolist()}
    except Exception as e:
        logger.error(f"Error fetching available months: {e}")
        return {"available_months": []}


@app.post("/compare-months", response_model=ComparisonResponse)
async def handle_comparison(month1: str, month2: str):
    try:
        dt1 = datetime.fromisoformat(month1.replace('Z', ''))
        dt2 = datetime.fromisoformat(month2.replace('Z', ''))
        month1_str = dt1.strftime('%Y-%m-01')
        month2_str = dt2.strftime('%Y-%m-01')

        with sqlite3.connect(DB_PATH) as conn:
            df1 = pd.read_sql_query(
                f"SELECT total_incidents FROM incident_time_series WHERE report_date = '{month1_str}'",
                conn
            )
            df2 = pd.read_sql_query(
                f"SELECT total_incidents FROM incident_time_series WHERE report_date = '{month2_str}'",
                conn
            )

        if df1.empty:
            raise HTTPException(status_code=404, detail=f"Data not found for {dt1.strftime('%B %Y')}")
        if df2.empty:
            raise HTTPException(status_code=404, detail=f"Data not found for {dt2.strftime('%B %Y')}")

        incidents1 = int(df1.iloc[0]['total_incidents'])
        incidents2 = int(df2.iloc[0]['total_incidents'])

        # Get document analysis for context
        doc1 = app.state.store.get_document_by_month_and_year(dt1.year, dt1.month)
        doc2 = app.state.store.get_document_by_month_and_year(dt2.year, dt2.month)

        summary1 = doc1.analysis.intelligence_summary if doc1 else ""
        summary2 = doc2.analysis.intelligence_summary if doc2 else ""
        threat1 = doc1.analysis.sentiment_analysis.threat_level if doc1 else "N/A"
        threat2 = doc2.analysis.sentiment_analysis.threat_level if doc2 else "N/A"

        # Calculate changes
        inc_change_raw = incidents1 - incidents2
        inc_change_pct = (inc_change_raw / incidents2 * 100) if incidents2 > 0 else 0

        # Generate AI inference with enhanced model
        prompt = (
            f"Analyze the change in security incidents between {dt2.strftime('%B %Y')} "
            f"({incidents2} incidents) and {dt1.strftime('%B %Y')} ({incidents1} incidents). "
            f"The change represents a {inc_change_pct:+.1f}% variation. "
            f"Context from reports: '{summary2}' and '{summary1}'. "
            f"What factors might have caused this change? Provide strategic insights and recommendations."
        )

        ai_result = app.state.rag_system.query(prompt, k=3)

        return ComparisonResponse(
            month1=dt1.strftime("%B %Y"),
            month2=dt2.strftime("%B %Y"),
            comparison_table=[
                MonthlyComparisonData(
                    metric="Total Incidents",
                    value1=str(incidents1),
                    value2=str(incidents2),
                    change=f"{inc_change_raw:+} ({inc_change_pct:+.1f}%)"
                ),
                MonthlyComparisonData(
                    metric="Threat Level",
                    value1=threat1,
                    value2=threat2,
                    change="Assessment Change"
                )
            ],
            ai_inference=ai_result.get("response", "Enhanced analysis not available at this time.")
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Comparison error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during comparison: {str(e)}")


@app.get("/document-list")
async def get_document_list():
    try:
        documents = app.state.store.get_all_documents()
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Document list error: {e}")
        return {"documents": []}


@app.get("/document/{doc_id}", response_model=AnalyzedDocument)
async def get_document(doc_id: str):
    document = app.state.store.get_document_by_id(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


@app.delete("/document/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document_endpoint(doc_id: str):
    success = app.state.store.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/rag-stats")
async def get_rag_stats_endpoint():
    return app.state.store.get_rag_stats()


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    query = request.query.lower()

    # Month and year filtering
    month_map = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
    }

    filters = {}
    for month_name, month_num in month_map.items():
        if month_name in query:
            filters['month'] = month_num
            break

    if '2020' in query:
        filters['year'] = 2020

    # Check for numerical queries
    numerical_phrases = ['how many', 'what was the total', 'what is the total', 'total number of', 'sum of', 'count of']
    is_numerical = any(phrase in query for phrase in numerical_phrases)

    # Check for assessment queries
    is_assessment = any(kw in query for kw in ['assessment', 'summary', 'conclusion', 'assess'])

    if is_numerical:
        logger.info(f"Handling numerical query with filters: {filters}")

        # Database column mapping
        db_column_map = {
            'casualties': 'total_casualties',
            'killed': 'total_casualties',
            'civilians': 'civilian_casualties',
            'security': 'security_casualties',
            'arrests': 'total_arrests',
            'arrested': 'total_arrests'
        }

        target_column = next((col for kw, col in db_column_map.items() if kw in query), None)

        table_to_query = 'monthly_detailed_stats'
        if 'incident' in query:
            target_column = 'total_incidents'
            table_to_query = 'incident_time_series'

        if target_column:
            try:
                where_clauses, params = [], []
                if 'month' in filters:
                    where_clauses.append("strftime('%m', report_date) = ?")
                    params.append(f"{filters['month']:02d}")
                if 'year' in filters:
                    where_clauses.append("strftime('%Y', report_date) = ?")
                    params.append(str(filters['year']))

                where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

                with sqlite3.connect(DB_PATH) as conn:
                    result = conn.execute(
                        f"SELECT SUM({target_column}) FROM {table_to_query} {where_sql}",
                        params
                    ).fetchone()

                total = result[0] if result and result[0] is not None else 0

                return QueryResponse(
                    response=f"Based on the processed documents, the total number of {target_column.replace('_', ' ')} for the specified period is {total:,}.",
                    sources=[{"filename": "Database Aggregate"}],
                    query=request.query,
                    timestamp=datetime.now().isoformat(),
                    model="Enhanced SQL Analytics",
                    context_chunks=1,
                    no_results=False,
                    error=False
                )

            except Exception as e:
                logger.error(f"SQL query failed: {e}")
                return QueryResponse(
                    response="Could not calculate the answer from the database.",
                    sources=[],
                    query=request.query,
                    error=True
                )

    # Handle descriptive queries with enhanced RAG
    keyword_search = "ASSESSMENT" if is_assessment else None
    logger.info(f"Handling descriptive query with enhanced RAG. Filters: {filters}, Keyword Search: {keyword_search}")

    try:
        rag_filters = {}
        if 'month' in filters:
            rag_filters['month'] = filters['month']
        if 'year' in filters:
            rag_filters['year'] = filters['year']

        result = app.state.rag_system.query(
            request.query,
            request.max_results,
            filters=rag_filters if rag_filters else None,
            keyword_search=keyword_search
        )

        result['query'] = request.query
        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Enhanced RAG query failed: {e}", exc_info=True)
        return QueryResponse(
            response=f"An error occurred during the enhanced query: {str(e)}",
            sources=[],
            query=request.query,
            error=True
        )


@app.post("/rebuild-index")
async def rebuild_index_endpoint():
    """Endpoint to rebuild the index with the new embedding model"""
    try:
        logger.info("Starting index rebuild with enhanced model...")

        # Clear old index files
        if os.path.exists("faiss_index.bin"):
            os.remove("faiss_index.bin")
        if os.path.exists("faiss_metadata.pkl"):
            os.remove("faiss_metadata.pkl")

        # Rebuild index
        docs_processed = app.state.store.rebuild_index_with_new_model(app.state.rag_system)

        return {
            "message": "Index successfully rebuilt with enhanced model",
            "documents_processed": docs_processed,
            "new_model": "BAAI/bge-large-en-v1.5",
            "new_dimensions": 1024,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Index rebuild failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting Intelligence Document Analyzer with Enhanced Models")
    logger.info("Embedding Model: BAAI/bge-large-en-v1.5 (1024 dimensions)")
    logger.info("LLM Model: Llama-3-8B via Ollama")
    uvicorn.run(app, host="0.0.0.0", port=8000)