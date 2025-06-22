"""
Intelligence Document Analyzer Backend - Complete with RAG Integration (FIXED VERSION)
FastAPI server for document analysis and intelligence extraction with RAG capabilities
"""

import os
import uuid
import time
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# NLP and Analysis Libraries
import spacy
import re
from collections import Counter, defaultdict
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize

# Document Processing Libraries
import PyPDF2
import docx
import pandas as pd
from io import StringIO, BytesIO

# RAG System Libraries
import numpy as np
import faiss
import openai
import json
import pickle
import tiktoken
from sentence_transformers import SentenceTransformer
import sqlite3
from pathlib import Path

# Load environment variables
load_dotenv()

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Intelligence Document Analyzer API with RAG",
    description="AI-Powered Security Intelligence Platform with Retrieval-Augmented Generation",
    version="4.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("SpaCy model loaded successfully")
except OSError:
    logger.error("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

# Initialize sentiment analyzer
try:
    sia = SentimentIntensityAnalyzer()
    logger.info("NLTK sentiment analyzer loaded successfully")
except:
    logger.error("NLTK sentiment analyzer failed to load")
    sia = None


# Data Models (Original)
class DocumentMetadata(BaseModel):
    filename: str
    file_type: str
    uploaded_at: str
    file_size: int


class DocumentClassification(BaseModel):
    primary_type: str
    sub_types: List[str]
    confidence: float
    security_classification: str


class SentimentAnalysis(BaseModel):
    overall_sentiment: str
    sentiment_score: float
    threat_level: str
    urgency_indicators: List[str]


class GeographicIntelligence(BaseModel):
    states: List[str]
    cities: List[str]
    countries: List[str]
    coordinates: List[Dict[str, Any]]
    total_locations: int
    other_locations: List[str]


class TemporalIntelligence(BaseModel):
    dates_mentioned: List[str]
    time_periods: List[str]
    months_mentioned: List[str]
    years_mentioned: List[str]
    temporal_patterns: List[str]


class NumericalIntelligence(BaseModel):
    incidents: List[int]
    casualties: List[int]
    weapons: List[int]
    arrests: List[int]
    monetary_values: List[float]


class CrimePatterns(BaseModel):
    primary_crimes: List[Tuple[str, int]]
    crime_frequency: Dict[str, int]
    crime_trends: List[Dict[str, Any]]


class TextStatistics(BaseModel):
    word_count: int
    sentence_count: int
    paragraph_count: int
    readability_score: float
    language: str


class DocumentAnalysis(BaseModel):
    document_classification: DocumentClassification
    entities: Dict[str, List[str]]
    sentiment_analysis: SentimentAnalysis
    geographic_intelligence: GeographicIntelligence
    temporal_intelligence: TemporalIntelligence
    numerical_intelligence: NumericalIntelligence
    crime_patterns: CrimePatterns
    relationships: List[Dict[str, Any]]
    text_statistics: TextStatistics
    intelligence_summary: str
    confidence_score: float
    processing_time: float


class AnalyzedDocument(BaseModel):
    id: str
    content: str
    metadata: DocumentMetadata
    analysis: DocumentAnalysis


# New RAG Models
class QueryRequest(BaseModel):
    query: str
    max_results: int = 5


class QueryResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    query: str
    context_chunks: int
    timestamp: str
    model: str
    error: Optional[bool] = None
    no_results: Optional[bool] = None


class RAGStats(BaseModel):
    total_chunks: int
    total_documents: int
    index_dimension: int
    model_name: str


# RAG System Classes (FIXED VERSION)
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
                           (document_id, chunk_text, chunk_index, embedding_id, metadata, filename)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ''', (
                               doc_id,
                               chunk['text'],
                               i,
                               chunk.get('embedding_id'),
                               json.dumps(chunk.get('metadata', {})),
                               chunk.get('metadata', {}).get('filename', f'doc_{doc_id}')
                           ))

        conn.commit()
        conn.close()

    def get_chunk_by_embedding_id(self, embedding_id: int) -> Optional[Dict]:
        """DEBUG VERSION - Find out what's really happening"""

        print(f"\n🔍 DEBUG get_chunk_by_embedding_id called")
        print(f"🔍 Looking for embedding_id: {embedding_id}")
        print(f"🔍 Database path: {self.db_path}")
        print(f"🔍 Database exists: {os.path.exists(self.db_path)}")
        print(f"🔍 Current working directory: {os.getcwd()}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Check database state at runtime
            cursor.execute("SELECT COUNT(*) FROM document_chunks")
            total_chunks = cursor.fetchone()[0]
            print(f"🔍 Total chunks in database: {total_chunks}")

            cursor.execute("SELECT COUNT(*) FROM document_chunks WHERE embedding_id = ?", (embedding_id,))
            found_count = cursor.fetchone()[0]
            print(f"🔍 Chunks with embedding_id {embedding_id}: {found_count}")

            cursor.execute(
                "SELECT MIN(embedding_id), MAX(embedding_id) FROM document_chunks WHERE embedding_id IS NOT NULL")
            min_max = cursor.fetchone()
            print(f"🔍 Runtime embedding_id range: {min_max[0]} to {min_max[1]}")

            # Show some sample embedding_ids
            cursor.execute("SELECT DISTINCT embedding_id FROM document_chunks ORDER BY embedding_id LIMIT 10")
            sample_ids = [row[0] for row in cursor.fetchall()]
            print(f"🔍 Sample embedding_ids: {sample_ids}")

            # Try the actual query
            cursor.execute('''
                           SELECT dc.chunk_text, dc.metadata, d.filename, d.metadata as doc_metadata
                           FROM document_chunks dc
                                    JOIN documents d ON dc.document_id = d.id
                           WHERE dc.embedding_id = ?
                           ''', (embedding_id,))

            result = cursor.fetchone()

            if result:
                print(f"🔍 SUCCESS: Found chunk for embedding_id {embedding_id}")
                print(f"🔍 Filename: {result[2]}")
                print(f"🔍 Text preview: {result[0][:100]}...")
            else:
                print(f"🔍 FAILED: No result from join query")

                # Try simple query
                cursor.execute("SELECT chunk_text FROM document_chunks WHERE embedding_id = ?", (embedding_id,))
                simple_result = cursor.fetchone()
                if simple_result:
                    print(f"🔍 BUT: Simple query works! Issue with JOIN")
                else:
                    print(f"🔍 CONFIRMED: embedding_id {embedding_id} truly not found")

        except Exception as e:
            print(f"🔍 EXCEPTION: {str(e)}")
            result = None

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
                'filename': result[2] if len(result) > 2 else f"unknown_doc_{embedding_id}",
                'document_metadata': doc_metadata
            }

        print(f"🔍 RETURNING None for embedding_id {embedding_id}")
        return None


class IntelligenceRAG:
    """Main RAG system for intelligence document analysis (FIXED VERSION)"""

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

    def chunk_text(self, text: str, max_tokens: int = 300, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks (IMPROVED VERSION)"""
        tokens = self.tokenizer.encode(text)
        chunks = []

        # If text is very short, return as single chunk
        if len(tokens) <= max_tokens:
            return [text]

        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Only add chunks with meaningful content
            if len(chunk_text.strip()) > 50:  # At least 50 characters
                chunks.append(chunk_text)

            if i + max_tokens >= len(tokens):
                break

        # Ensure we have at least one chunk
        if not chunks and text.strip():
            chunks = [text[:2000]]  # Fallback to first 2000 chars

        logger.info(f"Created {len(chunks)} chunks from {len(tokens)} tokens")
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
        """Retrieve most relevant chunks for a query (FIXED VERSION)"""
        if self.index.ntotal == 0:
            logger.warning("No documents in index")
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search for similar chunks - increase k to get more candidates
        search_k = min(k * 2, self.index.ntotal)  # Search more broadly
        similarities, indices = self.index.search(query_embedding.astype('float32'), search_k)

        logger.info(f"Search results for '{query}': similarities={similarities[0][:5]}, indices={indices[0][:5]}")

        relevant_chunks = []
        valid_results = 0

        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == -1:  # No more results
                break

            # Accept chunks with similarity > 0.1 (lowered threshold)
            if similarity > 0.1:
                logger.info(f"Found chunk at index {idx} with similarity {similarity:.4f}")

                chunk_data = self.document_store.get_chunk_by_embedding_id(idx)
                if chunk_data:
                    chunk_data['similarity'] = float(similarity)
                    chunk_data['rank'] = valid_results + 1
                    relevant_chunks.append(chunk_data)
                    valid_results += 1

                    # Stop when we have enough valid chunks
                    if valid_results >= k:
                        break
                else:
                    logger.warning(f"Could not retrieve chunk data for embedding_id {idx}")
            else:
                logger.info(f"Skipping chunk at index {idx} with low similarity {similarity:.4f}")

        logger.info(f"Successfully retrieved {len(relevant_chunks)} chunks with valid data")
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
                "context_chunks": len(relevant_chunks),
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-3.5-turbo-16k",
                "error": True
            }

    def query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Main query method - retrieve and generate (FIXED VERSION)"""
        logger.info(f"Processing query: {query}")

        # Debug: Check index stats
        stats = self.get_index_stats()
        logger.info(f"Current index stats: {stats}")

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
        return {
            "total_chunks": self.index.ntotal,
            "total_documents": len(
                set(meta['document_id'] for meta in self.index_metadata)) if self.index_metadata else 0,
            "index_dimension": self.dimension,
            "model_name": "all-MiniLM-L6-v2"
        }


# Intelligence Analysis Engine (Original - Simplified)
class IntelligenceAnalyzer:
    def __init__(self):
        self.crime_keywords = {
            'terrorism': ['terror', 'terrorist', 'bomb', 'explosive', 'attack', 'jihad', 'isis', 'al-qaeda'],
            'drug_trafficking': ['drug', 'cocaine', 'heroin', 'trafficking', 'smuggling', 'cartel', 'narcotic'],
            'human_trafficking': ['trafficking', 'slavery', 'prostitution', 'exploitation', 'forced labor'],
            'cybercrime': ['cyber', 'hacking', 'malware', 'phishing', 'ransomware', 'breach'],
            'organized_crime': ['mafia', 'gang', 'cartel', 'syndicate', 'racketeering'],
            'violence': ['murder', 'assault', 'kidnapping', 'robbery', 'violence', 'shooting'],
            'fraud': ['fraud', 'scam', 'embezzlement', 'money laundering', 'corruption'],
            'banditry': ['bandit', 'bandits', 'banditry', 'rustling', 'cattle rustling'],
            'armed_robbery': ['armed robbery', 'robbery', 'robber', 'robbers', 'armed robber']
        }

        self.weapon_keywords = [
            'gun', 'rifle', 'pistol', 'weapon', 'firearm', 'ammunition', 'explosive',
            'bomb', 'grenade', 'ak-47', 'ar-15', 'shotgun', 'revolver', 'ak 47',
            'pump action', 'locally made', 'double barrel'
        ]

        self.threat_indicators = [
            'threat', 'danger', 'attack', 'kill', 'destroy', 'eliminate', 'target',
            'strike', 'hit', 'bomb', 'explode', 'urgent', 'immediate', 'critical',
            'high threat', 'security threat', 'criminal activity'
        ]

    def extract_text_from_file(self, file_content: bytes, filename: str) -> str:
        """Extract text content from various file formats"""
        try:
            file_extension = filename.lower().split('.')[-1]

            if file_extension == 'pdf':
                return self._extract_from_pdf(file_content)
            elif file_extension in ['doc', 'docx']:
                return self._extract_from_docx(file_content)
            elif file_extension == 'txt':
                return file_content.decode('utf-8', errors='ignore')
            elif file_extension in ['csv', 'xlsx', 'xls']:
                return self._extract_from_excel(file_content)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

        except Exception as e:
            logger.error(f"Text extraction error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to extract text: {str(e)}")

    def _extract_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF files"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise ValueError(f"PDF extraction failed: {str(e)}")

    def _extract_from_docx(self, file_content: bytes) -> str:
        """Extract text from DOCX files"""
        try:
            doc = docx.Document(BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise ValueError(f"DOCX extraction failed: {str(e)}")

    def _extract_from_excel(self, file_content: bytes) -> str:
        """Extract text from Excel/CSV files"""
        try:
            # Try reading as Excel first
            try:
                df = pd.read_excel(BytesIO(file_content))
            except:
                # Fallback to CSV
                df = pd.read_csv(StringIO(file_content.decode('utf-8', errors='ignore')))

            # Convert DataFrame to text
            text = df.to_string()
            return text
        except Exception as e:
            raise ValueError(f"Excel/CSV extraction failed: {str(e)}")

    def analyze_document(self, text: str, metadata: DocumentMetadata) -> DocumentAnalysis:
        """Perform comprehensive intelligence analysis on document text"""
        start_time = time.time()

        # Basic text preprocessing
        text_lower = text.lower()
        sentences = sent_tokenize(text)
        words = word_tokenize(text_lower)

        # Extract entities using spaCy
        entities = self._extract_entities(text)

        # Analyze sentiment and threat level
        sentiment_analysis = self._analyze_sentiment(text, text_lower)

        # Extract geographic intelligence (simplified)
        geographic_intel = GeographicIntelligence(
            states=[], cities=[], countries=[], coordinates=[],
            total_locations=0, other_locations=[]
        )

        # Extract temporal intelligence (simplified)
        temporal_intel = TemporalIntelligence(
            dates_mentioned=[], time_periods=[], months_mentioned=[],
            years_mentioned=[], temporal_patterns=[]
        )

        # Extract numerical intelligence (simplified)
        numerical_intel = NumericalIntelligence(
            incidents=[], casualties=[], weapons=[], arrests=[], monetary_values=[]
        )

        # Analyze crime patterns
        crime_patterns = self._analyze_crime_patterns(text_lower)

        # Classify document
        doc_classification = DocumentClassification(
            primary_type="intelligence_report",
            sub_types=["security_analysis"],
            confidence=0.8,
            security_classification="RESTRICTED"
        )

        # Extract relationships (simplified)
        relationships = []

        # Calculate text statistics
        text_stats = TextStatistics(
            word_count=len(words),
            sentence_count=len(sentences),
            paragraph_count=len([p for p in text.split('\n\n') if p.strip()]),
            readability_score=50.0,
            language="en"
        )

        # Generate intelligence summary
        intelligence_summary = f"Intelligence analysis completed for {metadata.filename}. Document contains {len(words)} words with {sentiment_analysis.threat_level} threat level detected."

        # Calculate overall confidence score
        confidence_score = 0.75

        processing_time = time.time() - start_time

        return DocumentAnalysis(
            document_classification=doc_classification,
            entities=entities,
            sentiment_analysis=sentiment_analysis,
            geographic_intelligence=geographic_intel,
            temporal_intelligence=temporal_intel,
            numerical_intelligence=numerical_intel,
            crime_patterns=crime_patterns,
            relationships=relationships,
            text_statistics=text_stats,
            intelligence_summary=intelligence_summary,
            confidence_score=confidence_score,
            processing_time=processing_time
        )

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy NLP"""
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'weapons': [],
            'vehicles': [],
            'dates': []
        }

        if nlp is None:
            return entities

        try:
            doc = nlp(text)

            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities['persons'].append(ent.text)
                elif ent.label_ == "ORG":
                    entities['organizations'].append(ent.text)
                elif ent.label_ in ["GPE", "LOC"]:
                    entities['locations'].append(ent.text)
                elif ent.label_ == "DATE":
                    entities['dates'].append(ent.text)

            # Extract weapons using keyword matching
            text_lower = text.lower()
            for weapon in self.weapon_keywords:
                if weapon in text_lower:
                    entities['weapons'].append(weapon)

            # Remove duplicates and clean up
            for key in entities:
                entities[key] = list(set(entities[key]))

        except Exception as e:
            logger.error(f"Entity extraction error: {str(e)}")

        return entities

    def _analyze_sentiment(self, text: str, text_lower: str) -> SentimentAnalysis:
        """Analyze sentiment and determine threat level"""
        try:
            if sia:
                scores = sia.polarity_scores(text)
                sentiment_score = scores['compound']

                if sentiment_score >= 0.05:
                    overall_sentiment = "positive"
                elif sentiment_score <= -0.05:
                    overall_sentiment = "negative"
                else:
                    overall_sentiment = "neutral"
            else:
                overall_sentiment = "neutral"
                sentiment_score = 0.0

            # Determine threat level based on keywords
            threat_count = sum(1 for indicator in self.threat_indicators if indicator in text_lower)
            crime_indicators = ['armed robbery', 'murder', 'kidnapping', 'banditry', 'terrorism']
            crime_count = sum(1 for crime in crime_indicators if crime in text_lower)

            total_threat_score = threat_count + (crime_count * 2)

            if total_threat_score >= 8:
                threat_level = "High"
            elif total_threat_score >= 4:
                threat_level = "Medium"
            else:
                threat_level = "Low"

            urgency_indicators = [indicator for indicator in self.threat_indicators if indicator in text_lower]

            return SentimentAnalysis(
                overall_sentiment=overall_sentiment,
                sentiment_score=sentiment_score,
                threat_level=threat_level,
                urgency_indicators=list(set(urgency_indicators))[:10]
            )

        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return SentimentAnalysis(
                overall_sentiment="neutral",
                sentiment_score=0.0,
                threat_level="Low",
                urgency_indicators=[]
            )

    def _analyze_crime_patterns(self, text_lower: str) -> CrimePatterns:
        """Analyze crime patterns and frequencies"""
        crime_frequency = defaultdict(int)
        for crime_type, keywords in self.crime_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    crime_frequency[crime_type] += text_lower.count(keyword)

        primary_crimes = sorted(crime_frequency.items(), key=lambda x: x[1], reverse=True)[:5]

        return CrimePatterns(
            primary_crimes=primary_crimes,
            crime_frequency=dict(crime_frequency),
            crime_trends=[]
        )


# Initialize components
analyzer = IntelligenceAnalyzer()

# Initialize RAG system
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


# API Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Intelligence Document Analyzer API with RAG (FIXED)",
        "version": "4.0.1",
        "status": "operational",
        "rag_enabled": True,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        rag_system = get_rag_system()
        rag_stats = rag_system.get_index_stats()
        return {
            "status": "healthy",
            "spacy_model": nlp is not None,
            "nltk_sentiment": sia is not None,
            "rag_system": True,
            "openai_configured": bool(os.getenv('OPENAI_API_KEY')),
            "documents_indexed": rag_stats["total_documents"],
            "chunks_indexed": rag_stats["total_chunks"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.post("/upload-document", response_model=AnalyzedDocument)
async def upload_document(file: UploadFile = File(...)):
    """Upload and analyze intelligence document with RAG integration"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Read file content
        file_content = await file.read()
        file_size = len(file_content)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file provided")

        if file_size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large (max 50MB)")

        # Extract text from file
        try:
            text_content = analyzer.extract_text_from_file(file_content, file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract text: {str(e)}")

        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No readable text found in document")

        # Create metadata
        metadata = DocumentMetadata(
            filename=file.filename,
            file_type=file.filename.split('.')[-1].lower(),
            uploaded_at=datetime.now().isoformat(),
            file_size=file_size
        )

        # Perform intelligence analysis
        try:
            analysis = analyzer.analyze_document(text_content, metadata)
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

        # Generate document ID
        document_id = str(uuid.uuid4())

        # Add document to RAG system
        try:
            rag_system = get_rag_system()

            # Debug: Check current stats before adding
            current_stats = rag_system.get_index_stats()
            logger.info(f"RAG stats before adding document: {current_stats}")

            # Prepare metadata for RAG
            rag_metadata = {
                "filename": file.filename,
                "file_type": metadata.file_type,
                "uploaded_at": metadata.uploaded_at,
                "file_size": file_size,
                "analysis_summary": analysis.intelligence_summary,
                "threat_level": analysis.sentiment_analysis.threat_level,
                "confidence_score": analysis.confidence_score
            }

            chunks_added = rag_system.add_document(
                doc_id=document_id,
                filename=file.filename,
                content=text_content,
                metadata=rag_metadata
            )

            # Debug: Check stats after adding
            new_stats = rag_system.get_index_stats()
            logger.info(f"RAG stats after adding document: {new_stats}")
            logger.info(f"Added {chunks_added} chunks to RAG system for {file.filename}")

        except Exception as e:
            logger.error(f"RAG system error: {str(e)}")
            import traceback
            traceback.print_exc()
            # Continue even if RAG fails, but log the error
            chunks_added = 0

        # Create response (preserve first 2000 chars for display)
        preserved_content = text_content[:2000] + "..." if len(text_content) > 2000 else text_content

        analyzed_document = AnalyzedDocument(
            id=document_id,
            content=preserved_content,
            metadata=metadata,
            analysis=analysis
        )

        logger.info(f"Successfully analyzed document: {file.filename} (RAG chunks: {chunks_added})")
        return analyzed_document

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process AI query against document database using RAG"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="No query provided")

        # Get RAG system
        rag_system = get_rag_system()

        # Process query using RAG
        result = rag_system.query(request.query, k=request.max_results)

        return QueryResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/debug-rag")
async def debug_rag():
    """Debug endpoint to check RAG system status"""
    try:
        rag_system = get_rag_system()

        # Get basic stats
        stats = rag_system.get_index_stats()

        # Check database
        import sqlite3
        conn = sqlite3.connect('documents.db')
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM document_chunks")
        chunk_count = cursor.fetchone()[0]

        cursor.execute("SELECT filename, LENGTH(content) FROM documents LIMIT 5")
        sample_docs = cursor.fetchall()

        conn.close()

        # Test retrieval with simple query
        test_query = "security"
        chunks = rag_system.retrieve_relevant_chunks(test_query, k=3)

        return {
            "rag_stats": stats,
            "database": {
                "documents": doc_count,
                "chunks": chunk_count,
                "sample_docs": sample_docs
            },
            "test_retrieval": {
                "query": test_query,
                "chunks_found": len(chunks),
                "chunks": [{"similarity": c["similarity"], "filename": c["filename"], "text_preview": c["text"][:100]}
                           for c in chunks]
            },
            "index_info": {
                "total_vectors": rag_system.index.ntotal,
                "dimension": rag_system.index.d,
                "metadata_entries": len(rag_system.index_metadata)
            }
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.get("/rag-stats", response_model=RAGStats)
async def get_rag_stats():
    """Get RAG system statistics"""
    try:
        rag_system = get_rag_system()
        stats = rag_system.get_index_stats()
        return RAGStats(**stats)
    except Exception as e:
        logger.error(f"Error getting RAG stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get RAG stats: {str(e)}")


# Keep existing endpoints
@app.get("/document-list")
async def get_document_list():
    """Get list of processed documents"""
    mock_documents = [
        {
            "id": "doc_001",
            "filename": "intelligence_report_2025.pdf",
            "file_type": "pdf",
            "processed_at": "2025-06-21T10:30:00Z",
            "confidence_score": 0.92,
            "intelligence_summary": "High-priority intelligence report indicating increased security threats in Lagos region with multiple entities and locations identified.",
            "indexed_in_rag": True
        }
    ]
    return {"documents": mock_documents}


@app.get("/document/{document_id}")
async def get_document_details(document_id: str):
    """Get detailed analysis for a specific document"""
    raise HTTPException(status_code=404, detail="Document not found")


if __name__ == "__main__":
    # Ensure OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable not set!")
        exit(1)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )