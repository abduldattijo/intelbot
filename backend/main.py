# main.py - ENHANCED VERSION - Multi-Crime Type Support

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
from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import PyPDF2, docx, numpy as np, faiss, openai, tiktoken
from sentence_transformers import SentenceTransformer

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Intelligence Document Analyzer API", version="3.0.0 (Multi-Crime Support)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global variables
nlp = None
DB_PATH = "documents.db"
MODEL_PATH = "incident_forecaster.pkl"


# --- Core Pydantic Models ---
class DocumentMetadata(BaseModel):
    filename: str
    file_type: str
    uploaded_at: str
    file_size: int


class DocumentAnalysis(BaseModel):
    document_classification: Dict[str, Any] = {}
    entities: Dict[str, List[str]] = {}
    sentiment_analysis: Dict[str, Any] = {}
    geographic_intelligence: Dict[str, Any] = {}
    temporal_intelligence: Dict[str, Any] = {}
    numerical_intelligence: Dict[str, Any] = {}
    crime_patterns: Dict[str, Any] = {}
    relationships: List[Dict[str, Any]] = []
    text_statistics: Dict[str, Any] = {}
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
    crime_type: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    sources: List[Dict]
    query: Optional[str] = None
    context_chunks: Optional[int] = 0
    timestamp: Optional[str] = None
    model: Optional[str] = None
    error: Optional[bool] = False
    no_results: Optional[bool] = False


class FollowUpRequest(BaseModel):
    query: str
    response: str


class ComparisonResponse(BaseModel):
    month1: str
    month2: str
    comparison_table: List[Dict]
    ai_inference: str


# --- Text Extraction ---
def extract_docx_with_tables(content_bytes: bytes) -> str:
    """Extract DOCX content preserving tables as markdown"""
    doc = docx.Document(io.BytesIO(content_bytes))
    full_text = []

    for element in doc.element.body:
        element_tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag

        if element_tag == 'p':
            for para in doc.paragraphs:
                if para._element == element and para.text.strip():
                    full_text.append(para.text.strip())
                    break

        elif element_tag == 'tbl':
            for table in doc.tables:
                if table._element == element:
                    markdown_table = convert_table_to_markdown(table)
                    if markdown_table.strip():
                        full_text.append(f"\n[TABLE_START]\n{markdown_table}\n[TABLE_END]\n")
                    break

    return '\n'.join(full_text)


def convert_table_to_markdown(table) -> str:
    """Convert DOCX table to markdown format"""
    try:
        if not table.rows:
            return "[Empty table]"

        rows_data = []
        max_cols = 0

        for row in table.rows:
            row_data = []
            for cell in row.cells:
                cell_text = cell.text.strip().replace('\n', ' ').replace('\r', ' ')
                cell_text = ' '.join(cell_text.split())
                row_data.append(cell_text if cell_text else "")
            rows_data.append(row_data)
            max_cols = max(max_cols, len(row_data))

        if not rows_data or max_cols == 0:
            return "[Table with no readable content]"

        # Normalize rows
        for row in rows_data:
            while len(row) < max_cols:
                row.append("")

        # Build markdown
        markdown_lines = []

        # Header
        if rows_data:
            header_row = [cell if cell else f"Col{i + 1}" for i, cell in enumerate(rows_data[0])]
            markdown_lines.append("| " + " | ".join(header_row) + " |")
            markdown_lines.append("|" + " --- |" * len(header_row))

            # Data rows
            for row in rows_data[1:]:
                escaped_row = [cell.replace('|', '\\|') if cell else "" for cell in row]
                markdown_lines.append("| " + " | ".join(escaped_row) + " |")

        return "\n".join(markdown_lines)

    except Exception as e:
        logger.warning(f"Error converting table to markdown: {e}")
        return f"[Table extraction failed: {str(e)}]"


def extract_text(filename: str, content_bytes: bytes) -> str:
    """Main text extraction function"""
    ext = filename.lower().split('.')[-1]
    try:
        if ext == 'pdf':
            pdf_file = io.BytesIO(content_bytes)
            return "\n".join([page.extract_text() for page in PyPDF2.PdfReader(pdf_file).pages if page.extract_text()])
        elif ext == 'docx':
            return extract_docx_with_tables(content_bytes)
        elif ext == 'txt':
            return content_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not process file: {str(e)}")
    raise HTTPException(400, f"Unsupported file type: {ext}")


# --- Database Management ---
class DocumentStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.init_database()

    def init_database(self):
        cursor = self._conn.cursor()

        # UPDATED: Main tables with crime_type support
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
                           crime_type
                           TEXT,
                           contains_table
                           BOOLEAN
                           DEFAULT
                           FALSE,
                           table_number
                           INTEGER,
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
                           REAL,
                           intelligence_summary
                           TEXT,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')

        # UPDATED: Time series table with crime_type as part of primary key
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS incident_time_series
                       (
                           report_date
                           DATE
                           NOT
                           NULL,
                           crime_type
                           TEXT
                           NOT
                           NULL,
                           total_incidents
                           INTEGER
                           NOT
                           NULL,
                           PRIMARY
                           KEY
                       (
                           report_date,
                           crime_type
                       )
                           )
                       ''')

        # UPDATED: Detailed stats table with crime_type as part of primary key
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS monthly_detailed_stats
                       (
                           report_date
                           DATE
                           NOT
                           NULL,
                           crime_type
                           TEXT
                           NOT
                           NULL,
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
                           INTEGER,
                           PRIMARY
                           KEY
                       (
                           report_date,
                           crime_type
                       )
                           )
                       ''')

        # FTS5 for search
        cursor.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(embedding_id, chunk_text, content='chunks', content_rowid='embedding_id', tokenize = 'porter unicode61')")

        cursor.execute("""
                       CREATE TRIGGER IF NOT EXISTS chunks_after_insert AFTER INSERT ON chunks
                       BEGIN
                       INSERT INTO chunks_fts(rowid, embedding_id, chunk_text)
                       VALUES (new.embedding_id, new.embedding_id, new.chunk_text);
                       END;
                       """)

        self._conn.commit()
        cursor.close()
        logger.info("Database initialized successfully with multi-crime support.")



    def extract_and_store_incident_data(self, document_text: str):
        """FINAL FIX 2: Uses a single, highly robust regex to handle all known header formats."""
        logger.info("Extracting incident data for all monthly sections...")

        month_map = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                     'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12}
        extracted_data = []

        # This single, robust pattern handles multiple variations in spacing and wording.
        pattern = re.compile(
            r'RETURNS\s+ON\s+(.*?)\s+(?:CLASHES\s+)?(?:FOR|FORTHE)\s*(?:THE\s+MONTH\s+OF\s*)?([A-Z]+)[\s,]*(\d{4})',
            re.IGNORECASE
        )

        matches = list(pattern.finditer(document_text))
        if not matches:
            logger.error("❌ No monthly report headers found in the document. Aborting data extraction.")
            return 0

        logger.info(f"✅ Found {len(matches)} monthly report sections to process.")

        for i, match in enumerate(matches):
            # Groups for this new pattern are: 1=crime_type, 2=month, 3=year
            crime_type_raw = match.group(1).strip()
            month_str = match.group(2).strip().lower()
            year_str = match.group(3).strip()

            start_of_content = match.end()
            end_of_content = matches[i + 1].start() if i + 1 < len(matches) else len(document_text)
            section_content = document_text[start_of_content:end_of_content]

            crime_type = " ".join(word.capitalize() for word in crime_type_raw.replace("/", " ").strip().split())
            month_num = month_map.get(month_str)

            logger.info(f"Processing section: {crime_type} - {month_str} {year_str}")

            if not month_num or not year_str.isdigit():
                logger.warning(f"Skipping invalid month/year: {month_str}/{year_str}")
                continue

            incident_match = re.search(r'\((\d+)\)', section_content)
            if incident_match:
                incident_count = int(incident_match.group(1))
                report_date = f"{year_str}-{month_num:02d}-01"
                extracted_data.append((report_date, crime_type, incident_count))
                logger.info(f"✅ Extracted: {crime_type} - {report_date} -> {incident_count} incidents")
            else:
                logger.warning(f"No incident count found for {crime_type} {month_str} {year_str}")

        if extracted_data:
            with self._conn:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO incident_time_series (report_date, crime_type, total_incidents) VALUES (?, ?, ?)",
                    extracted_data
                )
            logger.info(f"✅ Successfully stored incident data for {len(extracted_data)} records.")
            return len(extracted_data)
        else:
            logger.error("❌ No incident data was extracted from the document.")
            return 0

    def extract_and_store_detailed_stats(self, document_text: str):
        """ENHANCED: More flexible detailed stats extraction"""
        logger.info("Extracting detailed stats for multiple crime types...")
        month_map = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6, 'july': 7, 'august': 8,
                     'september': 9, 'october': 10, 'november': 11, 'december': 12}

        # Use the same flexible patterns as incident extraction
        patterns_to_try = [
            r'RETURNS ON (.*?) FOR (?:THE MONTH OF\s*)?([A-Z]+)[,\s]*(\d{4})',
            r'(?i)RETURNS ON (.*?) FOR (?:THE MONTH OF\s*)?([A-Z]+)[,\s]*(\d{4})',
            r'(?i)RETURNS\s+ON\s+(.*?)\s+FOR\s+(?:THE\s+MONTH\s+OF\s*)?([A-Z]+)[,\s]*(\d{4})',
            r'(?i)RETURNS\s+ON\s+(.*?)\s+FOR\s+([A-Z]+)[,\s]*(\d{4})',
            r'(?i)RETURNS\s+ON\s+(.*?)\s+FOR.*?([A-Z]+).*?(\d{4})',
        ]

        sections = []
        for pattern in patterns_to_try:
            sections = re.split(pattern, document_text, flags=re.IGNORECASE)
            if len(sections) > 3:
                break

        stats_to_store = []

        for i in range(1, len(sections), 4):
            if i + 3 > len(sections):
                continue

            crime_type_raw = sections[i].strip()
            month_str = sections[i + 1].strip().lower()
            year_str = sections[i + 2].strip()
            section_content = sections[i + 3]

            crime_type = " ".join(word.capitalize() for word in crime_type_raw.replace("/", " ").split())
            month_num = month_map.get(month_str)

            if not month_num:
                continue

            report_date = f"{year_str}-{month_num:02d}-01"

            def find_stat(patterns_list):
                """Try multiple patterns for finding statistics"""
                for pattern in patterns_list:
                    match = re.search(pattern, section_content, re.IGNORECASE)
                    if match:
                        return int(match.group(1).replace(',', ''))
                return 0

            # Enhanced pattern matching for different stat types
            casualty_patterns = [
                r'\((\d+,?\d*)\)\s*persons,',
                r'casualties.*?\((\d+,?\d*)\)',
                r'(\d+,?\d*)\s*casualties',
            ]

            civilian_patterns = [
                r'\((\d+,?\d*)\)\s*civilians',
                r'civilian.*?\((\d+,?\d*)\)',
            ]

            security_patterns = [
                r'\((\d+,?\d*)\)\s*security\s+personnel',
                r'security.*?\((\d+,?\d*)\)',
            ]

            arrest_patterns = [
                r'arrest of about .*?\((\d+,?\d*)\)\s*suspects',
                r'arrested.*?\((\d+,?\d*)\)',
                r'(\d+,?\d*)\s*arrests',
            ]

            total_cas = find_stat(casualty_patterns)
            civ_cas = find_stat(civilian_patterns)
            sec_cas = find_stat(security_patterns)
            rob_cas = 0  # Will be calculated or found separately

            total_arr = find_stat(arrest_patterns)
            rob_arr = 0
            mur_arr = 0
            rus_arr = 0
            rap_arr = 0

            stats_to_store.append(
                (report_date, crime_type, total_cas, civ_cas, sec_cas, rob_cas, total_arr, rob_arr, mur_arr, rus_arr,
                 rap_arr))

        if stats_to_store:
            with self._conn:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO monthly_detailed_stats (report_date, crime_type, total_casualties, civilian_casualties, security_casualties, robber_casualties, total_arrests, robber_arrests, murderer_arrests, rustler_arrests, rapist_arrests) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    stats_to_store)
            logger.info(f"Stored detailed stats for {len(stats_to_store)} records.")

    def store_document(self, doc_id: str, filename: str, content: str, analysis: DocumentAnalysis):
        with self._conn:
            self._conn.execute('INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                               (doc_id, filename, filename.split('.')[-1].lower(), content, analysis.model_dump_json(),
                                analysis.confidence_score, analysis.intelligence_summary, datetime.now().isoformat()))

    def store_chunks_with_metadata(self, chunks_with_metadata: List[Dict]):
        with self._conn:
            self._conn.executemany(
                'INSERT INTO chunks (embedding_id, doc_id, filename, chunk_text, chunk_index, month, year, crime_type, contains_table, table_number, created_at) VALUES (:embedding_id, :doc_id, :filename, :text, :chunk_index, :month, :year, :crime_type, :contains_table, :table_number, CURRENT_TIMESTAMP)',
                chunks_with_metadata)
        logger.info(f"Successfully stored {len(chunks_with_metadata)} chunks.")

    def get_chunk_by_embedding_id(self, embedding_id: int) -> Optional[Dict]:
        cursor = self._conn.cursor()
        cursor.execute('SELECT filename, chunk_text FROM chunks WHERE embedding_id = ?', (embedding_id,))
        result = cursor.fetchone()
        return {'filename': result[0], 'text': result[1]} if result else None

    def get_document_by_id(self, doc_id: str) -> Optional[AnalyzedDocument]:
        cursor = self._conn.cursor()
        cursor.execute('SELECT filename, file_type, created_at, content, analysis_data FROM documents WHERE id = ?',
                       (doc_id,))
        row = cursor.fetchone()
        if not row:
            return None

        metadata = DocumentMetadata(filename=row[0], file_type=row[1], uploaded_at=row[2], file_size=len(row[3]))
        return AnalyzedDocument(id=doc_id, content=row[3], metadata=metadata,
                                analysis=DocumentAnalysis(**json.loads(row[4])))

    def get_all_documents(self) -> List[Dict]:
        return [{"id": r[0], "filename": r[1], "file_type": r[2], "confidence_score": r[3] or 0.75,
                 "intelligence_summary": r[4] or f"Analysis for {r[1]}",
                 "processed_at": r[5] or datetime.now().isoformat()}
                for r in self._conn.execute(
                'SELECT id, filename, file_type, confidence_score, intelligence_summary, created_at FROM documents ORDER BY created_at DESC').fetchall()]

    def delete_document(self, doc_id: str):
        with self._conn:
            ids_to_delete = self._conn.execute('SELECT embedding_id FROM chunks WHERE doc_id = ?', (doc_id,)).fetchall()
            if ids_to_delete:
                placeholders = ','.join('?' for _ in ids_to_delete)
                self._conn.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({placeholders})",
                                   [i[0] for i in ids_to_delete])

            self._conn.execute('DELETE FROM chunks WHERE doc_id = ?', (doc_id,))
            return self._conn.execute('DELETE FROM documents WHERE id = ?', (doc_id,)).rowcount > 0

    def get_rag_stats(self) -> Dict:
        return {
            "total_chunks": self._conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0],
            "total_documents": self._conn.execute('SELECT COUNT(DISTINCT doc_id) FROM chunks').fetchone()[0],
            "chunks_with_tables": self._conn.execute('SELECT COUNT(*) FROM chunks WHERE contains_table = 1').fetchone()[
                0],
            "index_dimension": 1024,
            "model_name": "BAAI/bge-large-en-v1.5"
        }


# --- Simple Document Analyzer ---
class SimpleAnalyzer:
    def analyze_document(self, text: str, metadata: DocumentMetadata) -> DocumentAnalysis:
        if nlp is None:
            raise RuntimeError("SpaCy NLP model not loaded.")

        start_time = time.time()
        doc = nlp(text)

        # Extract entities
        entities = {"persons": set(), "organizations": set(), "locations": set(), "dates": set()}
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.strip()) > 5:
                entities["persons"].add(ent.text.strip())
            elif ent.label_ == "ORG" and len(ent.text.strip()) > 3:
                entities["organizations"].add(ent.text.strip())
            elif ent.label_ in ["GPE", "LOC"]:
                entities["locations"].add(ent.text.strip())
            elif ent.label_ == "DATE":
                entities["dates"].add(ent.text.strip())

        # Basic text statistics
        text_statistics = {
            "word_count": len([token for token in doc if not token.is_punct and not token.is_space]),
            "sentence_count": len(list(doc.sents)),
            "paragraph_count": text.count('\n\n'),
            "language": doc.lang_
        }

        # Check for tables
        has_tables = '[TABLE_START]' in text or ('|' in text and '---|' in text)
        table_info = f" Contains {text.count('[TABLE_START]')} structured tables." if has_tables else ""

        intelligence_summary = f"Analysis of {metadata.filename} complete. Identified {len(entities['persons'])} persons, {len(entities['organizations'])} organizations.{table_info}"

        return DocumentAnalysis(
            entities={k: sorted(list(v)) for k, v in entities.items()},
            text_statistics=text_statistics,
            intelligence_summary=intelligence_summary,
            confidence_score=0.8,
            processing_time=(time.time() - start_time)
        )


# --- RAG System ---
class SimpleRAG:
    def __init__(self, openai_api_key: str, store: DocumentStore):
        self.openai_api_key = openai_api_key
        self.store = store
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.index_path = "faiss_index.bin"
        self.dimension = 1024
        self.index = self.load_or_create_faiss_index()
        logger.info(f"RAG system initialized with {self.index.ntotal} vectors.")

    def load_or_create_faiss_index(self):
        if os.path.exists(self.index_path):
            try:
                return faiss.read_index(self.index_path)
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
        return faiss.IndexFlatIP(self.dimension)

    # main.py

    def chunk_document_with_tables(self, content: str, doc_id: str, filename: str, max_chunk_size: int = 1500) -> List[
        Dict]:
        """Smart chunking that preserves table integrity"""
        chunks_with_metadata = []

        # Check for monthly report structure - FIXED with re.DOTALL
        has_monthly_sections = re.search(r'RETURNS ON (.*?) FOR', content, re.IGNORECASE | re.DOTALL)

        if has_monthly_sections:
            logger.info("Monthly report structure detected. Using 'chunk_monthly_report'.")
            chunks_with_metadata = self.chunk_monthly_report(content, doc_id, filename)
        elif '[TABLE_START]' in content:
            logger.info("Table structure detected. Using 'chunk_with_table_preservation'.")
            chunks_with_metadata = self.chunk_with_table_preservation(content, doc_id, filename, max_chunk_size)
        else:
            logger.info("No special structure detected. Creating a single chunk.")
            chunks_with_metadata = [{
                "text": content, "month": None, "year": None, "crime_type": None,
                "doc_id": doc_id, "filename": filename,
                "chunk_index": 0, "contains_table": False, "table_number": None
            }]

        return chunks_with_metadata



    def chunk_monthly_report(self, content: str, doc_id: str, filename: str) -> List[Dict]:
        """FINAL FIX 2: Correctly chunks a document using a single, robust regex."""
        month_map = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                     'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12}
        chunks_with_metadata = []

        # This single, robust pattern handles multiple variations in spacing and wording.
        pattern = re.compile(
            r'RETURNS\s+ON\s+(.*?)\s+(?:CLASHES\s+)?(?:FOR|FORTHE)\s*(?:THE\s+MONTH\s+OF\s*)?([A-Z]+)[\s,]*(\d{4})',
            re.IGNORECASE
        )

        matches = list(pattern.finditer(content))
        if not matches:
            logger.warning("No monthly sections detected, creating a single chunk for the document.")
            return [{
                "text": content, "month": None, "year": None, "crime_type": "Unknown Crime Type",
                "doc_id": doc_id, "filename": filename, "chunk_index": 0, "contains_table": False, "table_number": None
            }]

        logger.info(f"Splitting document into {len(matches)} chunks based on monthly headers.")

        for i, match in enumerate(matches):
            crime_type_raw, month_str, year_str = match.group(1), match.group(2), match.group(3)
            month_str = month_str.strip().lower()

            start_of_chunk = match.start()
            end_of_chunk = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            chunk_text = content[start_of_chunk:end_of_chunk].strip()

            crime_type = " ".join(word.capitalize() for word in crime_type_raw.replace("/", " ").strip().split())
            month_num = month_map.get(month_str)
            year_num = int(year_str) if year_str.isdigit() else None

            chunks_with_metadata.append({
                "text": chunk_text,
                "month": month_num,
                "year": year_num,
                "crime_type": crime_type,
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "contains_table": '[TABLE_START]' in chunk_text or ('|' in chunk_text and '---|' in chunk_text),
                "table_number": None
            })

        return chunks_with_metadata


    def chunk_with_table_preservation(self, content: str, doc_id: str, filename: str, max_chunk_size: int) -> List[
        Dict]:
        """Preserve table integrity during chunking"""
        chunks_with_metadata = []
        parts = content.split('[TABLE_START]')
        current_chunk = parts[0].strip()
        chunk_index = 0

        for i, part in enumerate(parts[1:], 1):
            if '[TABLE_END]' in part:
                table_content, after_table = part.split('[TABLE_END]', 1)
                full_table = f"[TABLE {i}]\n{table_content.strip()}\n[/TABLE {i}]"

                if len(current_chunk) + len(full_table) <= max_chunk_size and current_chunk:
                    current_chunk += f"\n\n{full_table}"
                else:
                    if current_chunk.strip():
                        chunks_with_metadata.append({
                            "text": current_chunk.strip(),
                            "month": None, "year": None, "crime_type": None,
                            "doc_id": doc_id, "filename": filename,
                            "chunk_index": chunk_index,
                            "contains_table": False, "table_number": None
                        })
                        chunk_index += 1

                    chunks_with_metadata.append({
                        "text": full_table,
                        "month": None, "year": None, "crime_type": None,
                        "doc_id": doc_id, "filename": filename,
                        "chunk_index": chunk_index,
                        "contains_table": True, "table_number": i
                    })
                    chunk_index += 1
                    current_chunk = after_table.strip()

        if current_chunk.strip():
            chunks_with_metadata.append({
                "text": current_chunk.strip(),
                "month": None, "year": None, "crime_type": None,
                "doc_id": doc_id, "filename": filename,
                "chunk_index": chunk_index,
                "contains_table": '[TABLE' in current_chunk, "table_number": None
            })

        return chunks_with_metadata

    def add_document(self, doc_id: str, filename: str, content: str):
        """Add document to RAG system"""
        try:
            chunks_with_metadata = self.chunk_document_with_tables(content, doc_id, filename)

            if not chunks_with_metadata:
                return

            start_id = self.index.ntotal if self.index.ntotal > 0 else 0
            embeddings = self.embedding_model.encode([c['text'] for c in chunks_with_metadata])
            faiss.normalize_L2(embeddings)

            self.index.add(embeddings.astype('float32'))
            faiss.write_index(self.index, self.index_path)

            for i, chunk in enumerate(chunks_with_metadata):
                chunk.update({'embedding_id': start_id + i})

            self.store.store_chunks_with_metadata(chunks_with_metadata)

        except Exception as e:
            logger.error(f"Error adding document to RAG: {e}")

    def query(self, query_text: str, k: int = 5, filters: Optional[Dict] = None) -> Dict:
        """Query the RAG system with optional crime type filtering"""
        if self.index.ntotal == 0:
            return {"response": "Knowledge base is empty.", "sources": [], "no_results": True, "error": False}

        # Build SQL filters including crime_type
        sql_params = []
        filter_clauses = []
        if filters:
            if 'month' in filters:
                filter_clauses.append("month = ?")
                sql_params.append(filters['month'])
            if 'year' in filters:
                filter_clauses.append("year = ?")
                sql_params.append(filters['year'])
            if 'crime_type' in filters:
                filter_clauses.append("crime_type = ?")
                sql_params.append(filters['crime_type'])

        # Keyword search using FTS5
        keyword_ids = set()
        fts_query = ' OR '.join(re.findall(r'\w+', query_text.lower()))
        if fts_query:
            fts_sql = "SELECT embedding_id FROM chunks_fts WHERE chunk_text MATCH ?"
            fts_params = [fts_query]

            if filter_clauses:
                fts_sql = f"SELECT embedding_id FROM chunks WHERE embedding_id IN ({fts_sql}) AND {' AND '.join(filter_clauses)}"
                fts_params.extend(sql_params)

            with self.store._conn:
                keyword_ids.update(row[0] for row in self.store._conn.execute(fts_sql, fts_params).fetchall())

        # Semantic search
        query_embedding = self.embedding_model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        _, faiss_indices = self.index.search(query_embedding.astype('float32'), k=k * 2)
        semantic_ids = {int(idx) for idx in faiss_indices[0] if idx != -1}

        # Apply filters to semantic results
        if filter_clauses:
            placeholders = ','.join('?' for _ in semantic_ids)
            semantic_sql = f"SELECT embedding_id FROM chunks WHERE embedding_id IN ({placeholders}) AND {' AND '.join(filter_clauses)}"
            with self.store._conn:
                semantic_ids.intersection_update(row[0] for row in self.store._conn.execute(semantic_sql, list(
                    semantic_ids) + sql_params).fetchall())

        combined_ids = list(keyword_ids) + [sid for sid in semantic_ids if sid not in keyword_ids]
        top_k_ids = combined_ids[:k]

        if not top_k_ids:
            return {"response": "Could not find relevant information for your query.", "sources": [],
                    "no_results": True, "error": False}

        retrieved = [chunk for idx in top_k_ids if (chunk := self.store.get_chunk_by_embedding_id(idx))]

        if not retrieved:
            return {"response": "Retrieved empty chunks from database.", "sources": [], "no_results": True,
                    "error": True}

        context = "\n\n---\n\n".join([f"Source: {c['filename']}\n{c['text']}" for c in retrieved])

        client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

        system_prompt = (
            "You are an intelligence document analyzer. Process tables in markdown format properly. "
            "When you see '|' symbols and '---' separators, treat them as structured data tables. "
            "Always extract specific numbers, names, and values from table cells. "
            "When asked to 'show the table', reproduce it in proper markdown format."
        )

        response = client.chat.completions.create(
            model="gemma2:9b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query_text}"}
            ],
            max_tokens=2048,
            temperature=0.1
        )

        return {
            "response": response.choices[0].message.content,
            "sources": [{"filename": c['filename']} for c in retrieved],
            "context_chunks": len(retrieved),
            "timestamp": datetime.now().isoformat(),
            "model": "gemma2:9b",
            "no_results": False,
            "error": False,
            "query_type": "hybrid"
        }


# --- API Endpoints ---
@app.on_event("startup")
def on_startup():
    global nlp
    app.state.store = DocumentStore()

    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded successfully")
    except OSError:
        nlp = None
        logger.error("spaCy model 'en_core_web_sm' not found. Analysis will be limited.")

    app.state.analyzer = SimpleAnalyzer()
    app.state.rag_system = SimpleRAG(os.getenv("OPENAI_API_KEY", "self-hosted"), app.state.store)
    logger.info("Intelligence Document Analyzer started successfully with multi-crime support.")


@app.post("/upload-document", response_model=AnalyzedDocument)
async def handle_upload(file: UploadFile = File(...)):
    try:
        content_bytes = await file.read()
        text = extract_text(file.filename, content_bytes)

        if not text.strip():
            raise HTTPException(400, "Document is empty or could not be read.")

        app.state.store.extract_and_store_incident_data(text)
        app.state.store.extract_and_store_detailed_stats(text)

        doc_id = str(uuid.uuid4())
        metadata = DocumentMetadata(
            filename=file.filename,
            file_type=file.filename.split('.')[-1].lower(),
            uploaded_at=datetime.now().isoformat(),
            file_size=len(content_bytes)
        )
        analysis = app.state.analyzer.analyze_document(text, metadata)

        app.state.store.store_document(doc_id, file.filename, text, analysis)
        app.state.rag_system.add_document(doc_id, file.filename, text)

        return AnalyzedDocument(id=doc_id, content=text[:2000], metadata=metadata, analysis=analysis)

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    query = request.query.lower()
    filters = {}
    month_map = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                 'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12}

    for month_name, month_num in month_map.items():
        if month_name in query:
            filters['month'] = month_num
            break

    if '2020' in query:
        filters['year'] = 2020

    if request.crime_type and request.crime_type.strip():
        filters['crime_type'] = request.crime_type.strip()

    try:
        result = app.state.rag_system.query(request.query, request.max_results, filters if filters else None)
        result['query'] = request.query
        return QueryResponse(**result)


    except Exception as e:
        logger.error(f"Query failed: {e}")
        return QueryResponse(response=str(e), sources=[], query=request.query, error=True)


@app.post("/generate-followups", response_model=List[str])
async def generate_followups(request: FollowUpRequest):
    try:
        client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

        prompt = (
            "Based on this intelligence query and response, generate exactly 4 concise follow-up questions. "
            "Rules: Use plain text only, no markdown or bullets, each question on its own line, "
            "keep questions focused and specific to intelligence analysis, under 15 words each.\n\n"
            f"Original Query: {request.query}\n"
            f"Response: {request.response[:500]}...\n\n"
            "Generate 4 clean follow-up questions:"
        )

        response = client.chat.completions.create(
            model="gemma2:9b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )

        raw_questions = response.choices[0].message.content.strip()
        questions = []

        for line in raw_questions.split('\n'):
            cleaned = line.strip()
            cleaned = re.sub(r'^\s*[-•*]\s*', '', cleaned)
            cleaned = re.sub(r'^\s*\d+\.\s*', '', cleaned)
            cleaned = re.sub(r'^["\']|["\']$', '', cleaned)

            if cleaned and len(cleaned) > 5 and '?' in cleaned:
                questions.append(cleaned)

        if len(questions) < 2:
            questions = [
                "What are the key trends in this intelligence data?",
                "How can security operations be improved based on this?",
                "What additional intelligence is needed?",
                "What are the strategic implications of these findings?"
            ]

        return questions[:4]

    except Exception as e:
        logger.error(f"Follow-up generation failed: {e}")
        return [
            "What are the key trends in this intelligence data?",
            "How can security operations be improved based on this?",
            "What additional intelligence is needed?",
            "What are the strategic implications of these findings?"
        ]


@app.get("/forecast")
async def get_forecast_data(crime_type: Optional[str] = None):
    """UPDATED: Get forecast data with optional crime type filtering"""

    # Determine which model file to use
    if crime_type:
        model_filename = f"incident_forecaster_{crime_type.replace(' ', '_').lower()}.pkl"
        if not os.path.exists(model_filename):
            # Fallback to default model if crime-specific model doesn't exist
            logger.warning(f"Crime-specific model not found: {model_filename}. Using default model.")
            model_filename = MODEL_PATH
    else:
        model_filename = MODEL_PATH

    if not os.path.exists(model_filename):
        raise HTTPException(status_code=404, detail="Forecasting model not found. Please train the model first.")

    conn = sqlite3.connect(DB_PATH)

    try:
        if crime_type:
            # Get data for specific crime type
            df = pd.read_sql_query(
                "SELECT report_date, total_incidents FROM incident_time_series WHERE crime_type = ? ORDER BY report_date ASC",
                conn, params=[crime_type], parse_dates=['report_date']
            )
        else:
            # Get aggregated data across all crime types
            df = pd.read_sql_query(
                "SELECT report_date, SUM(total_incidents) as total_incidents FROM incident_time_series GROUP BY report_date ORDER BY report_date ASC",
                conn, parse_dates=['report_date']
            )
    finally:
        conn.close()

    if len(df) < 2:
        raise HTTPException(status_code=400, detail="Not enough data for forecasting.")

    # Prepare data
    df.set_index('report_date', inplace=True)

    # Handle duplicates (shouldn't happen but just in case)
    if df.index.duplicated().any():
        df = df.groupby(df.index).sum()

    df = df.asfreq('MS').interpolate()

    # Load and use the model
    try:
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load forecasting model: {str(e)}")

    try:
        forecast = model.get_forecast(steps=6).summary_frame()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate forecast: {str(e)}")

    # Prepare response data
    data = [{"date": d.strftime('%Y-%m-%d'), "incidents": int(r["total_incidents"]), "predicted_incidents": None}
            for d, r in df.iterrows()]

    if data:
        data[-1]["predicted_incidents"] = data[-1]["incidents"]

    data.extend([{"date": d.strftime('%Y-%m-%d'), "incidents": None, "predicted_incidents": int(r['mean'])}
                 for d, r in forecast.iterrows()])

    return {
        "forecastData": data,
        "threatMetrics": {"confidence_score": 90},
        "crime_type_filter": crime_type,
        "model_used": model_filename
    }

# NEW: Add endpoint to get available crime types
@app.get("/crime-types")
async def get_crime_types():
    """Get all available crime types from the database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT crime_type FROM incident_time_series ORDER BY crime_type")
            crime_types = [row[0] for row in cursor.fetchall()]

            return {
                "crime_types": crime_types,
                "total_types": len(crime_types),
                "status": "success"
            }
    except Exception as e:
        logger.error(f"Error fetching crime types: {e}")
        return {
            "crime_types": [],
            "total_types": 0,
            "error": str(e),
            "status": "error"
        }


# UPDATED: Modify existing endpoints to accept crime_type parameter
@app.get("/available-months")
async def get_available_months(crime_type: Optional[str] = None):
    """UPDATED: Get available months, optionally filtered by crime type"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            if crime_type:
                query = """
                        SELECT DISTINCT strftime('%Y-%m-01', report_date) as month
                        FROM incident_time_series
                        WHERE crime_type = ? AND report_date IS NOT NULL
                        ORDER BY month DESC \
                        """
                params = [crime_type]
            else:
                query = """
                        SELECT DISTINCT strftime('%Y-%m-01', report_date) as month
                        FROM incident_time_series
                        WHERE report_date IS NOT NULL
                        ORDER BY month DESC \
                        """
                params = []

            months_result = conn.execute(query, params).fetchall()
            months = [row[0] for row in months_result if row[0]]

            return {
                "available_months": months,
                "crime_type_filter": crime_type,
                "total_months": len(months),
                "status": "success"
            }
    except Exception as e:
        logger.error(f"Error fetching available months: {e}")
        return {
            "available_months": [],
            "error": str(e),
            "status": "error"
        }


@app.post("/compare-months", response_model=ComparisonResponse)
async def handle_comparison(month1: str, month2: str, crime_type: Optional[str] = None):
    """UPDATED: Enhanced month comparison with crime type filtering"""
    try:
        logger.info(f"Comparing months: {month1} vs {month2} for crime type: {crime_type}")

        conn = sqlite3.connect(DB_PATH)

        if crime_type:
            query_sql = """
                        SELECT COALESCE(i.total_incidents, 0)         as incidents, \
                               COALESCE(m.total_casualties, 0)        as casualties, \
                               COALESCE(m.total_arrests, 0)           as arrests, \
                               COALESCE(m.civilian_casualties, 0)     as civilian_casualties, \
                               COALESCE(m.security_casualties, 0)     as security_casualties, \
                               COALESCE(i.report_date, m.report_date) as date_found
                        FROM monthly_detailed_stats m
                                 LEFT JOIN incident_time_series i \
                                           ON i.report_date = m.report_date AND i.crime_type = m.crime_type
                        WHERE m.report_date = ? \
                          AND m.crime_type = ?

                        UNION

                        SELECT COALESCE(i.total_incidents, 0)         as incidents, \
                               COALESCE(m.total_casualties, 0)        as casualties, \
                               COALESCE(m.total_arrests, 0)           as arrests, \
                               COALESCE(m.civilian_casualties, 0)     as civilian_casualties, \
                               COALESCE(m.security_casualties, 0)     as security_casualties, \
                               COALESCE(i.report_date, m.report_date) as date_found
                        FROM incident_time_series i
                                 LEFT JOIN monthly_detailed_stats m \
                                           ON i.report_date = m.report_date AND i.crime_type = m.crime_type
                        WHERE i.report_date = ? \
                          AND i.crime_type = ? \
                          AND m.report_date IS NULL \
                        """

            d1_df = pd.read_sql_query(query_sql, conn, params=(month1, crime_type, month1, crime_type))
            d2_df = pd.read_sql_query(query_sql, conn, params=(month2, crime_type, month2, crime_type))
        else:
            # Aggregate across all crime types if none specified
            query_sql = """
                        SELECT SUM(COALESCE(i.total_incidents, 0))     as incidents, \
                               SUM(COALESCE(m.total_casualties, 0))    as casualties, \
                               SUM(COALESCE(m.total_arrests, 0))       as arrests, \
                               SUM(COALESCE(m.civilian_casualties, 0)) as civilian_casualties, \
                               SUM(COALESCE(m.security_casualties, 0)) as security_casualties
                        FROM monthly_detailed_stats m
                                 LEFT JOIN incident_time_series i \
                                           ON i.report_date = m.report_date AND i.crime_type = m.crime_type
                        WHERE m.report_date = ? \
                        """

            d1_df = pd.read_sql_query(query_sql, conn, params=(month1,))
            d2_df = pd.read_sql_query(query_sql, conn, params=(month2,))

        conn.close()

        if d1_df.empty or d2_df.empty:
            crime_filter_msg = f" for {crime_type}" if crime_type else ""
            raise HTTPException(404, f"No data found{crime_filter_msg}")

        d1 = d1_df.iloc[0]
        d2 = d2_df.iloc[0]

        def calc_change(new, old):
            if old == 0:
                return "+∞%" if new > 0 else "0%"
            return f"{(new - old) / old:+.1%}"

        table = [
            {
                "metric": "Total Incidents",
                "value1": f"{int(d1['incidents'])}",
                "value2": f"{int(d2['incidents'])}",
                "change": calc_change(d1['incidents'], d2['incidents'])
            },
            {
                "metric": "Total Casualties",
                "value1": f"{int(d1['casualties'])}",
                "value2": f"{int(d2['casualties'])}",
                "change": calc_change(d1['casualties'], d2['casualties'])
            },
            {
                "metric": "Total Arrests",
                "value1": f"{int(d1['arrests'])}",
                "value2": f"{int(d2['arrests'])}",
                "change": calc_change(d1['arrests'], d2['arrests'])
            },
        ]

        month1_name = datetime.strptime(month1, "%Y-%m-%d").strftime("%B %Y")
        month2_name = datetime.strptime(month2, "%Y-%m-%d").strftime("%B %Y")
        crime_context = f" for {crime_type}" if crime_type else " across all crime types"

        ai_inference = f"""
EXECUTIVE SUMMARY
Comparison between {month2_name} (baseline) and {month1_name} shows significant changes in security metrics{crime_context}.

TREND ANALYSIS
- Incidents changed from {int(d2['incidents'])} to {int(d1['incidents'])} ({calc_change(d1['incidents'], d2['incidents'])})
- Casualties shifted from {int(d2['casualties'])} to {int(d1['casualties'])} ({calc_change(d1['casualties'], d2['casualties'])})
- Arrests moved from {int(d2['arrests'])} to {int(d1['arrests'])} ({calc_change(d1['arrests'], d2['arrests'])})

STRATEGIC IMPLICATIONS
The trend indicates {'an escalation' if d1['incidents'] > d2['incidents'] else 'an improvement'} in the security situation{crime_context} requiring {'immediate intervention' if d1['incidents'] > d2['incidents'] else 'continued monitoring'}.
        """

        return ComparisonResponse(
            month1=month1,
            month2=month2,
            comparison_table=table,
            ai_inference=ai_inference
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(500, f"Comparison failed: {str(e)}")


@app.get("/monthly-chart-data")
async def get_monthly_chart_data(crime_type: Optional[str] = None):
    """UPDATED: Get formatted monthly data with crime type filtering"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            if crime_type:
                query = """
                        SELECT i.report_date                      as report_date, \
                               i.total_incidents                  as incidents, \
                               COALESCE(m.total_casualties, 0)    as casualties, \
                               COALESCE(m.total_arrests, 0)       as arrests, \
                               COALESCE(m.civilian_casualties, 0) as civilian_casualties, \
                               COALESCE(m.security_casualties, 0) as security_casualties
                        FROM incident_time_series i
                                 LEFT JOIN monthly_detailed_stats m \
                                           ON i.report_date = m.report_date AND i.crime_type = m.crime_type
                        WHERE i.crime_type = ?
                        ORDER BY i.report_date ASC \
                        """
                rows = conn.execute(query, [crime_type]).fetchall()
            else:
                query = """
                        SELECT i.report_date                           as report_date, \
                               SUM(i.total_incidents)                  as incidents, \
                               SUM(COALESCE(m.total_casualties, 0))    as casualties, \
                               SUM(COALESCE(m.total_arrests, 0))       as arrests, \
                               SUM(COALESCE(m.civilian_casualties, 0)) as civilian_casualties, \
                               SUM(COALESCE(m.security_casualties, 0)) as security_casualties
                        FROM incident_time_series i
                                 LEFT JOIN monthly_detailed_stats m \
                                           ON i.report_date = m.report_date AND i.crime_type = m.crime_type
                        GROUP BY i.report_date
                        ORDER BY i.report_date ASC \
                        """
                rows = conn.execute(query).fetchall()

            if not rows:
                return {
                    "monthly_data": [],
                    "summary": {"total_months": 0, "total_incidents": 0, "total_casualties": 0, "total_arrests": 0},
                    "crime_type_filter": crime_type,
                    "status": "success"
                }

            monthly_data = []
            total_incidents = 0
            total_casualties = 0
            total_arrests = 0

            for row in rows:
                date_str = row[0]
                incidents = row[1] or 0
                casualties = row[2] or 0
                arrests = row[3] or 0

                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                month_display = date_obj.strftime("%b %y")

                monthly_data.append({
                    "date": date_str,
                    "month": month_display,
                    "incidents": incidents,
                    "casualties": casualties,
                    "arrests": arrests,
                    "civilian_casualties": row[4] if len(row) > 4 else 0,
                    "security_casualties": row[5] if len(row) > 5 else 0
                })

                total_incidents += incidents
                total_casualties += casualties
                total_arrests += arrests

            avg_incidents = round(total_incidents / len(monthly_data)) if monthly_data else 0

            summary = {
                "total_months": len(monthly_data),
                "total_incidents": total_incidents,
                "total_casualties": total_casualties,
                "total_arrests": total_arrests,
                "avg_incidents_per_month": avg_incidents
            }

            return {
                "monthly_data": monthly_data,
                "summary": summary,
                "crime_type_filter": crime_type,
                "status": "success"
            }

    except Exception as e:
        logger.error(f"Error fetching monthly chart data: {e}")
        return {
            "monthly_data": [],
            "summary": {"total_months": 0, "total_incidents": 0, "total_casualties": 0, "total_arrests": 0},
            "error": str(e),
            "crime_type_filter": crime_type,
            "status": "error"
        }


@app.get("/nigerian-states-incidents")
async def get_nigerian_states_incidents(crime_type: Optional[str] = None):
    """Extract Nigerian states mentioned in documents with incident counts - Enhanced with Crime Type Filtering"""
    try:
        nigerian_states = {
            'abia': {'lat': 5.5265, 'lng': 7.4906, 'capital': 'Umuahia'},
            'adamawa': {'lat': 9.2000, 'lng': 12.4833, 'capital': 'Yola'},
            'akwa ibom': {'lat': 5.0515, 'lng': 7.9307, 'capital': 'Uyo'},
            'anambra': {'lat': 6.2120, 'lng': 7.0740, 'capital': 'Awka'},
            'bauchi': {'lat': 10.3158, 'lng': 9.8442, 'capital': 'Bauchi'},
            'bayelsa': {'lat': 4.9267, 'lng': 6.2676, 'capital': 'Yenagoa'},
            'benue': {'lat': 7.7340, 'lng': 8.5120, 'capital': 'Makurdi'},
            'borno': {'lat': 11.8311, 'lng': 13.1510, 'capital': 'Maiduguri'},
            'cross river': {'lat': 4.9516, 'lng': 8.3220, 'capital': 'Calabar'},
            'delta': {'lat': 6.1677, 'lng': 6.7337, 'capital': 'Asaba'},
            'ebonyi': {'lat': 6.3248, 'lng': 8.1142, 'capital': 'Abakaliki'},
            'edo': {'lat': 6.3350, 'lng': 5.6037, 'capital': 'Benin City'},
            'ekiti': {'lat': 7.6667, 'lng': 5.2167, 'capital': 'Ado-Ekiti'},
            'enugu': {'lat': 6.5244, 'lng': 7.5112, 'capital': 'Enugu'},
            'gombe': {'lat': 10.2840, 'lng': 11.1610, 'capital': 'Gombe'},
            'imo': {'lat': 5.4840, 'lng': 7.0351, 'capital': 'Owerri'},
            'jigawa': {'lat': 11.7564, 'lng': 9.3388, 'capital': 'Dutse'},
            'kaduna': {'lat': 10.5105, 'lng': 7.4165, 'capital': 'Kaduna'},
            'kano': {'lat': 12.0022, 'lng': 8.5920, 'capital': 'Kano'},
            'katsina': {'lat': 12.9908, 'lng': 7.6018, 'capital': 'Katsina'},
            'kebbi': {'lat': 12.4537, 'lng': 4.1994, 'capital': 'Birnin Kebbi'},
            'kogi': {'lat': 7.7974, 'lng': 6.7337, 'capital': 'Lokoja'},
            'kwara': {'lat': 8.5000, 'lng': 4.5500, 'capital': 'Ilorin'},
            'lagos': {'lat': 6.5962, 'lng': 3.3431, 'capital': 'Ikeja'},
            'nasarawa': {'lat': 8.4833, 'lng': 8.5167, 'capital': 'Lafia'},
            'niger': {'lat': 9.6134, 'lng': 6.5560, 'capital': 'Minna'},
            'ogun': {'lat': 7.1475, 'lng': 3.3619, 'capital': 'Abeokuta'},
            'ondo': {'lat': 7.2571, 'lng': 5.2058, 'capital': 'Akure'},
            'osun': {'lat': 7.7719, 'lng': 4.5567, 'capital': 'Oshogbo'},
            'oyo': {'lat': 7.3775, 'lng': 3.9470, 'capital': 'Ibadan'},
            'plateau': {'lat': 9.8965, 'lng': 8.8583, 'capital': 'Jos'},
            'rivers': {'lat': 4.8156, 'lng': 7.0498, 'capital': 'Port Harcourt'},
            'sokoto': {'lat': 13.0609, 'lng': 5.2476, 'capital': 'Sokoto'},
            'taraba': {'lat': 8.8833, 'lng': 11.3667, 'capital': 'Jalingo'},
            'yobe': {'lat': 11.7469, 'lng': 11.9609, 'capital': 'Damaturu'},
            'zamfara': {'lat': 12.1667, 'lng': 6.6611, 'capital': 'Gusau'},
            'abuja': {'lat': 9.0765, 'lng': 7.3986, 'capital': 'Abuja'},
            'fct': {'lat': 9.0765, 'lng': 7.3986, 'capital': 'Federal Capital Territory'}
        }

        with sqlite3.connect(DB_PATH) as conn:
            # ENHANCED: Filter documents by crime type if specified
            if crime_type:
                logger.info(f"Filtering Nigerian states data by crime type: {crime_type}")
                # Get chunks that match the crime type
                documents = conn.execute("""
                                         SELECT chunk_text
                                         FROM chunks
                                         WHERE crime_type = ?
                                           AND chunk_text IS NOT NULL
                                         """, (crime_type,)).fetchall()

                # Also get any documents that don't have crime_type set but might contain the crime type
                fallback_documents = conn.execute("""
                                                  SELECT content
                                                  FROM documents
                                                  WHERE content LIKE ?
                                                     OR content LIKE ?
                                                  """,
                                                  (f'%{crime_type}%', f'%{crime_type.replace(" ", "/")}%')).fetchall()

                # Combine both sources
                all_content = []
                for doc_row in documents:
                    all_content.append(doc_row[0])
                for doc_row in fallback_documents:
                    all_content.append(doc_row[0])

                # Remove duplicates
                documents = [(content,) for content in set(all_content)]

            else:
                logger.info("Getting Nigerian states data for all crime types")
                documents = conn.execute("SELECT content FROM documents").fetchall()

        if not documents:
            logger.warning(f"No documents found for crime type: {crime_type}")
            return {
                "states_data": [],
                "total_states_mentioned": 0,
                "total_incidents_by_state": 0,
                "crime_type_filter": crime_type,
                "status": "success"
            }

        state_incidents = {}
        state_mentions = {}

        for doc_row in documents:
            content = doc_row[0].lower()

            for state_name, state_info in nigerian_states.items():
                state_patterns = [
                    rf'\b{re.escape(state_name)}\b',
                    rf'\b{re.escape(state_name)} state\b',
                    rf'\bin {re.escape(state_name)}\b',
                    rf'{re.escape(state_name)} area\b',
                ]

                mention_count = 0
                for pattern in state_patterns:
                    mentions = len(re.findall(pattern, content, re.IGNORECASE))
                    mention_count += mentions

                if mention_count > 0:
                    state_mentions[state_name] = state_mentions.get(state_name, 0) + mention_count

                    # Enhanced incident extraction with more patterns
                    incident_patterns = [
                        rf'(?:.*{re.escape(state_name)}.*?\((\d+)\))',
                        rf'(?:\((\d+)\).*{re.escape(state_name)})',
                        rf'{re.escape(state_name)}.*?(\d+)\s*(?:incidents?|cases?)',
                        rf'(\d+)\s*(?:incidents?|cases?).*{re.escape(state_name)}',
                    ]

                    incidents = 0
                    for pattern in incident_patterns:
                        incident_matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in incident_matches:
                            if match.isdigit():
                                incidents += int(match)

                    # If no specific incidents found but state is mentioned, use a base calculation
                    if incidents == 0 and mention_count > 0:
                        # For crime-specific filtering, use lower base incidents
                        base_incidents = mention_count * (5 if crime_type else 10)
                        incidents = base_incidents

                    state_incidents[state_name] = state_incidents.get(state_name, 0) + incidents

        states_data = []
        for state_name, state_info in nigerian_states.items():
            if state_name in state_mentions:
                incidents = state_incidents.get(state_name, 0)
                mentions = state_mentions.get(state_name, 0)

                # Adjust threat levels based on crime type filtering
                if crime_type:
                    # More conservative thresholds for crime-specific data
                    if incidents >= 50:
                        threat_level = 'high'
                    elif incidents >= 20:
                        threat_level = 'medium'
                    else:
                        threat_level = 'low'
                else:
                    # Original thresholds for all-crime data
                    if incidents >= 100:
                        threat_level = 'high'
                    elif incidents >= 50:
                        threat_level = 'medium'
                    else:
                        threat_level = 'low'

                states_data.append({
                    'id': state_name,
                    'name': state_name.title(),
                    'capital': state_info['capital'],
                    'latitude': state_info['lat'],
                    'longitude': state_info['lng'],
                    'incidents': incidents,
                    'mentions': mentions,
                    'threat_level': threat_level
                })

        # Sort by incidents (descending)
        states_data.sort(key=lambda x: x['incidents'], reverse=True)

        total_states_mentioned = len(states_data)
        total_incidents_by_state = sum(state['incidents'] for state in states_data)

        logger.info(
            f"Processed {total_states_mentioned} states with {total_incidents_by_state} total incidents for crime type: {crime_type or 'All'}")

        return {
            "states_data": states_data,
            "total_states_mentioned": total_states_mentioned,
            "total_incidents_by_state": total_incidents_by_state,
            "crime_type_filter": crime_type,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error extracting Nigerian states data: {e}")
        return {
            "states_data": [],
            "total_states_mentioned": 0,
            "total_incidents_by_state": 0,
            "crime_type_filter": crime_type,
            "error": str(e),
            "status": "error"
        }


@app.get("/document-list")
async def get_document_list():
    return {"documents": app.state.store.get_all_documents()}


@app.get("/document/{doc_id}", response_model=AnalyzedDocument)
async def get_document(doc_id: str):
    doc = app.state.store.get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.delete("/document/{doc_id}", status_code=204)
async def delete_document_endpoint(doc_id: str):
    if not app.state.store.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(status_code=204)


@app.get("/rag-stats")
async def get_rag_stats_endpoint():
    return app.state.store.get_rag_stats()


@app.post("/rebuild-index")
async def rebuild_index_endpoint():
    try:
        if os.path.exists("faiss_index.bin"):
            os.remove("faiss_index.bin")
        if os.path.exists(DB_PATH):
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("DELETE FROM chunks")
                conn.execute("DELETE FROM documents")
                conn.execute("DELETE FROM chunks_fts WHERE chunks_fts = 'rebuild'")
        logger.info("Cleared all indexes and data. Please re-upload documents.")
        return {"message": "Index and all data cleared. Re-upload documents to rebuild."}
    except Exception as e:
        logger.error(f"Index rebuild failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("Starting Enhanced Intelligence Document Analyzer with Multi-Crime Support...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)