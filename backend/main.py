# main.py - Intelligence Document Processor

import os
import io
import re
import json
import spacy
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import numpy as np

# File processing imports
import PyPDF2
import pdfplumber
from docx import Document
import pytesseract
from PIL import Image
import easyocr

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Machine Learning imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import nltk
from collections import Counter

app = FastAPI(title="Intelligence Document Analyzer", version="2.0.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Initialize OCR
easyocr_reader = easyocr.Reader(['en'])


class DocumentProcessor:
    """Comprehensive document processing for intelligence analysis"""

    def __init__(self):
        self.processed_documents = {}
        self.intelligence_patterns = self._load_intelligence_patterns()

    def _load_intelligence_patterns(self) -> Dict[str, List[str]]:
        """Define patterns for extracting intelligence data"""
        return {
            'incidents': [
                r'(\d+)\s*(?:incidents?|cases?|attacks?|events?)',
                r'total\s+of\s+(\d+)',
                r'(\d+)\s*criminal\s+activities?'
            ],
            'casualties': [
                r'(\d+)\s*(?:casualties|fatalities|deaths?|killed)',
                r'(\d+)\s*(?:persons?|people)\s+(?:lost\s+their\s+lives?|died|killed)',
                r'(\d+)\s*lives?\s+lost'
            ],
            'weapons': [
                r'(\d+)\s*(?:AK-?47s?|rifles?|guns?|weapons?)',
                r'(\d+)\s*(?:pistols?|firearms?|ammunition)',
                r'recovered\s+(\d+)'
            ],
            'arrests': [
                r'(\d+)\s*(?:arrests?|suspects?|apprehended)',
                r'(\d+)\s*(?:criminals?|perpetrators?)\s+(?:arrested|caught)'
            ],
            'locations': [
                r'(?:in|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:State|LGA|Local|Area)',
                r'([A-Z][a-z]+)\s+State',
                r'(?:Abuja|FCT)'
            ]
        }

    async def process_document(self, file: UploadFile) -> Dict[str, Any]:
        """Main document processing function"""
        try:
            file_content = await file.read()
            file_extension = file.filename.split('.')[-1].lower()

            # Extract text based on file type
            if file_extension == 'pdf':
                extracted_text = await self._extract_from_pdf(file_content)
            elif file_extension in ['doc', 'docx']:
                extracted_text = await self._extract_from_word(file_content)
            elif file_extension == 'txt':
                extracted_text = file_content.decode('utf-8', errors='ignore')
            elif file_extension in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']:
                extracted_text = await self._extract_from_image_ocr(file_content)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Analyze the extracted text
            analysis_result = await self._analyze_intelligence_text(
                extracted_text, file.filename, file_extension
            )

            # Store processed document
            doc_id = f"{file.filename}_{datetime.now().isoformat()}"
            self.processed_documents[doc_id] = {
                'original_filename': file.filename,
                'file_type': file_extension,
                'extracted_text': extracted_text,
                'analysis': analysis_result,
                'processed_at': datetime.now().isoformat()
            }

            return {
                'document_id': doc_id,
                'status': 'success',
                'analysis': analysis_result,
                'metadata': {
                    'filename': file.filename,
                    'file_type': file_extension,
                    'text_length': len(extracted_text),
                    'processed_at': datetime.now().isoformat()
                }
            }

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing document: {str(e)}")

    async def _extract_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF files"""
        text = ""

        try:
            # Try with pdfplumber first (better for complex PDFs)
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except:
            # Fallback to PyPDF2
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            except:
                # If PDF is scanned/image-based, try OCR
                return await self._extract_from_pdf_ocr(file_content)

        return text if text.strip() else await self._extract_from_pdf_ocr(file_content)

    async def _extract_from_word(self, file_content: bytes) -> str:
        """Extract text from Word documents"""
        try:
            doc = Document(io.BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"

            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                    text += "\n"

            return text
        except Exception as e:
            raise ValueError(f"Error extracting from Word document: {str(e)}")

    async def _extract_from_image_ocr(self, file_content: bytes) -> str:
        """Extract text from images using OCR"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(file_content))

            # Try EasyOCR first
            try:
                results = easyocr_reader.readtext(np.array(image))
                text = " ".join([result[1] for result in results])
                return text
            except:
                # Fallback to Tesseract
                text = pytesseract.image_to_string(image)
                return text

        except Exception as e:
            raise ValueError(f"Error in OCR processing: {str(e)}")

    async def _extract_from_pdf_ocr(self, file_content: bytes) -> str:
        """Extract text from PDF using OCR (for scanned PDFs)"""
        try:
            import pdf2image
            pages = pdf2image.convert_from_bytes(file_content)
            text = ""

            for page in pages:
                # Convert PIL image to numpy array for EasyOCR
                results = easyocr_reader.readtext(np.array(page))
                page_text = " ".join([result[1] for result in results])
                text += page_text + "\n"

            return text
        except Exception as e:
            # Fallback to simpler OCR
            return f"OCR extraction failed: {str(e)}"

    async def _analyze_intelligence_text(self, text: str, filename: str, file_type: str) -> Dict[str, Any]:
        """Comprehensive intelligence analysis of extracted text"""

        # Basic text statistics
        word_count = len(text.split())
        sentence_count = len(re.findall(r'[.!?]+', text))

        # Extract numerical intelligence data
        numerical_data = self._extract_numerical_intelligence(text)

        # Extract locations
        locations = self._extract_locations(text)

        # Extract dates and time references
        temporal_data = self._extract_temporal_references(text)

        # Sentiment analysis
        sentiment = self._analyze_sentiment(text)

        # Extract key entities
        entities = self._extract_entities(text)

        # Identify document type and classification
        doc_classification = self._classify_document(text, filename)

        # Extract crime patterns
        crime_patterns = self._extract_crime_patterns(text)

        # Generate intelligence summary
        intelligence_summary = self._generate_intelligence_summary(
            numerical_data, locations, temporal_data, crime_patterns
        )

        return {
            'document_classification': doc_classification,
            'text_statistics': {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'character_count': len(text)
            },
            'numerical_intelligence': numerical_data,
            'geographic_intelligence': locations,
            'temporal_intelligence': temporal_data,
            'sentiment_analysis': sentiment,
            'entities': entities,
            'crime_patterns': crime_patterns,
            'intelligence_summary': intelligence_summary,
            'confidence_score': self._calculate_confidence_score(text, numerical_data)
        }

    def _extract_numerical_intelligence(self, text: str) -> Dict[str, List[int]]:
        """Extract numerical data using intelligence patterns"""
        results = {}

        for category, patterns in self.intelligence_patterns.items():
            numbers = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                numbers.extend([int(match) for match in matches if match.isdigit()])
            results[category] = sorted(set(numbers), reverse=True)  # Remove duplicates, sort desc

        return results

    def _extract_locations(self, text: str) -> Dict[str, Any]:
        """Extract geographic intelligence"""
        doc = nlp(text)

        # Nigerian states and major cities
        nigerian_locations = {
            'states': [
                'Zamfara', 'Katsina', 'Kaduna', 'Plateau', 'Niger', 'Kano', 'Sokoto',
                'Kebbi', 'Jigawa', 'Bauchi', 'Gombe', 'Adamawa', 'Taraba', 'Borno',
                'Yobe', 'Nasarawa', 'Benue', 'Kogi', 'Lagos', 'Ogun', 'Oyo', 'Osun',
                'Ekiti', 'Ondo', 'Delta', 'Edo', 'Rivers', 'Bayelsa', 'Cross River',
                'Akwa Ibom', 'Abia', 'Imo', 'Anambra', 'Enugu', 'Ebonyi'
            ],
            'zones': [
                'North-West', 'North-Central', 'North-East',
                'South-West', 'South-East', 'South-South'
            ]
        }

        found_locations = {'states': [], 'zones': [], 'other_locations': []}

        # Extract using spaCy NER
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                location = ent.text.strip()
                if location in nigerian_locations['states']:
                    found_locations['states'].append(location)
                elif location in nigerian_locations['zones']:
                    found_locations['zones'].append(location)
                else:
                    found_locations['other_locations'].append(location)

        # Extract using regex patterns
        for pattern in self.intelligence_patterns['locations']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match in nigerian_locations['states']:
                    found_locations['states'].append(match)

        # Remove duplicates and return counts
        return {
            'states': list(set(found_locations['states'])),
            'zones': list(set(found_locations['zones'])),
            'other_locations': list(set(found_locations['other_locations'])),
            'total_locations': len(set(found_locations['states'] + found_locations['zones']))
        }

    def _extract_temporal_references(self, text: str) -> Dict[str, Any]:
        """Extract time-based intelligence"""

        # Month patterns
        months = re.findall(
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', text,
            re.IGNORECASE)

        # Year patterns
        years = re.findall(r'\b(20\d{2})\b', text)

        # Time periods
        time_periods = re.findall(r'\b(morning|evening|night|dawn|dusk|midnight)\b', text, re.IGNORECASE)

        return {
            'months_mentioned': list(set([m.lower().capitalize() for m in months])),
            'years_mentioned': list(set(years)),
            'time_periods': list(set([t.lower() for t in time_periods])),
            'temporal_density': len(months) + len(years) + len(time_periods)
        }

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment and urgency of the document"""
        blob = TextBlob(text)

        # Threat level indicators
        threat_words = ['critical', 'urgent', 'emergency', 'immediate', 'severe', 'high', 'escalating']
        threat_count = sum(1 for word in threat_words if word in text.lower())

        return {
            'polarity': blob.sentiment.polarity,  # -1 (negative) to 1 (positive)
            'subjectivity': blob.sentiment.subjectivity,  # 0 (objective) to 1 (subjective)
            'threat_level': 'High' if threat_count > 3 else 'Medium' if threat_count > 1 else 'Low',
            'threat_indicators': threat_count
        }

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities and key terms"""
        doc = nlp(text)

        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'money': [],
            'weapons': []
        }

        # Extract using spaCy NER
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities['persons'].append(ent.text)
            elif ent.label_ == "ORG":
                entities['organizations'].append(ent.text)
            elif ent.label_ in ["GPE", "LOC"]:
                entities['locations'].append(ent.text)
            elif ent.label_ == "DATE":
                entities['dates'].append(ent.text)
            elif ent.label_ == "MONEY":
                entities['money'].append(ent.text)

        # Extract weapons mentions
        weapon_patterns = r'\b(AK-?47|rifle|pistol|gun|weapon|ammunition|explosive|IED)\b'
        weapons = re.findall(weapon_patterns, text, re.IGNORECASE)
        entities['weapons'] = list(set(weapons))

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities

    def _classify_document(self, text: str, filename: str) -> Dict[str, Any]:
        """Classify the document type and extract metadata"""

        # Document type classification
        doc_types = {
            'intelligence_report': ['intelligence', 'report', 'analysis', 'assessment'],
            'incident_report': ['incident', 'occurrence', 'event', 'case'],
            'security_briefing': ['briefing', 'update', 'situation', 'sitrep'],
            'operational_report': ['operation', 'ops', 'mission', 'deployment']
        }

        classification_scores = {}
        text_lower = text.lower()

        for doc_type, keywords in doc_types.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            classification_scores[doc_type] = score

        # Determine primary classification
        primary_type = max(classification_scores, key=classification_scores.get)

        # Extract security classification
        security_patterns = r'\b(SECRET|CONFIDENTIAL|RESTRICTED|CLASSIFIED|UNCLASSIFIED)\b'
        security_classifications = re.findall(security_patterns, text, re.IGNORECASE)

        return {
            'primary_type': primary_type,
            'confidence': classification_scores[primary_type] / len(doc_types[primary_type]) if doc_types[
                primary_type] else 0,
            'security_classification': security_classifications[0] if security_classifications else 'UNCLASSIFIED',
            'classification_scores': classification_scores
        }

    def _extract_crime_patterns(self, text: str) -> Dict[str, Any]:
        """Extract crime patterns and modus operandi"""

        crime_types = {
            'armed_robbery': r'\b(armed\s+robbery|robbery|banditry)\b',
            'cattle_rustling': r'\b(cattle\s+rustling|livestock|animals?\s+rustled)\b',
            'kidnapping': r'\b(kidnapping|abduction|hostage)\b',
            'murder': r'\b(murder|killing|homicide|assassination)\b',
            'terrorism': r'\b(terrorism|terrorist|insurgency|extremist)\b'
        }

        patterns = {}
        for crime, pattern in crime_types.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 0:
                patterns[crime] = matches

        # Extract modus operandi keywords
        mo_keywords = re.findall(r'\b(motorcycle|forest|highway|checkpoint|raid|attack|escape)\b', text, re.IGNORECASE)

        return {
            'crime_frequencies': patterns,
            'primary_crimes': sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:3],
            'modus_operandi_indicators': list(set([m.lower() for m in mo_keywords]))
        }

    def _generate_intelligence_summary(self, numerical_data: Dict, locations: Dict,
                                       temporal_data: Dict, crime_patterns: Dict) -> str:
        """Generate an AI-powered intelligence summary"""

        summary_parts = []

        # Incident summary
        if numerical_data.get('incidents'):
            max_incidents = max(numerical_data['incidents'])
            summary_parts.append(f"Document reports {max_incidents} incidents")

        # Casualty summary
        if numerical_data.get('casualties'):
            max_casualties = max(numerical_data['casualties'])
            summary_parts.append(f"{max_casualties} casualties recorded")

        # Geographic summary
        if locations['states']:
            primary_states = ', '.join(locations['states'][:3])
            summary_parts.append(f"Primary affected areas: {primary_states}")

        # Crime summary
        if crime_patterns['primary_crimes']:
            primary_crime = crime_patterns['primary_crimes'][0][0]
            summary_parts.append(f"Dominant crime type: {primary_crime.replace('_', ' ').title()}")

        # Temporal summary
        if temporal_data['months_mentioned']:
            months = ', '.join(temporal_data['months_mentioned'][:2])
            summary_parts.append(f"Time frame: {months}")

        return ". ".join(summary_parts) + "." if summary_parts else "Limited intelligence data extracted."

    def _calculate_confidence_score(self, text: str, numerical_data: Dict) -> float:
        """Calculate confidence score for the extraction"""

        score = 0.0

        # Text quality indicators
        if len(text) > 500:
            score += 0.2
        if len(re.findall(r'\d+', text)) > 5:
            score += 0.2

        # Data extraction success
        if any(numerical_data.values()):
            score += 0.3

        # Structure indicators
        if any(word in text.lower() for word in ['report', 'analysis', 'summary']):
            score += 0.2

        # Date/time references
        if re.findall(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',
                      text, re.IGNORECASE):
            score += 0.1

        return min(score, 1.0)


# Initialize processor
processor = DocumentProcessor()


@app.post("/upload-document")
async def upload_document(
        file: UploadFile = File(...),
        analysis_type: str = Form("full")
):
    """Upload and analyze documents in multiple formats"""

    # Validate file type
    allowed_extensions = ['pdf', 'doc', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'tiff', 'bmp']
    file_extension = file.filename.split('.')[-1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    try:
        result = await processor.process_document(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-documents")
async def query_documents(query_data: Dict[str, str]):
    """Query processed documents using natural language"""

    query = query_data.get("query", "").lower()

    if not processor.processed_documents:
        return {"response": "No documents have been processed yet. Please upload a document first."}

    # Simple query processing (can be enhanced with vector search)
    response = "Based on processed documents:\n\n"

    # Aggregate data from all processed documents
    total_incidents = 0
    total_casualties = 0
    all_locations = set()
    all_crimes = Counter()

    for doc_id, doc_data in processor.processed_documents.items():
        analysis = doc_data['analysis']

        # Aggregate numerical data
        if analysis['numerical_intelligence'].get('incidents'):
            total_incidents += max(analysis['numerical_intelligence']['incidents'])
        if analysis['numerical_intelligence'].get('casualties'):
            total_casualties += max(analysis['numerical_intelligence']['casualties'])

        # Aggregate locations
        all_locations.update(analysis['geographic_intelligence']['states'])

        # Aggregate crimes
        for crime, count in analysis['crime_patterns']['crime_frequencies'].items():
            all_crimes[crime] += count

    if 'summary' in query or 'overview' in query:
        response += f"📊 **Document Analysis Summary**\n"
        response += f"• Total documents processed: {len(processor.processed_documents)}\n"
        response += f"• Total incidents identified: {total_incidents}\n"
        response += f"• Total casualties recorded: {total_casualties}\n"
        response += f"• Affected locations: {', '.join(list(all_locations)[:5])}\n"
        response += f"• Primary crime types: {', '.join([crime.replace('_', ' ').title() for crime, _ in all_crimes.most_common(3)])}\n"

    elif 'location' in query or 'geographic' in query:
        response += f"🗺️ **Geographic Intelligence**\n"
        response += f"• Affected states: {', '.join(all_locations)}\n"
        response += f"• Total locations mentioned: {len(all_locations)}\n"

    elif 'crime' in query or 'pattern' in query:
        response += f"🎯 **Crime Pattern Analysis**\n"
        for crime, count in all_crimes.most_common(5):
            response += f"• {crime.replace('_', ' ').title()}: {count} mentions\n"

    else:
        response += "Available queries: 'summary', 'location analysis', 'crime patterns'\n"
        response += f"Currently tracking {len(processor.processed_documents)} processed documents."

    return {"response": response}


@app.get("/document-list")
async def get_document_list():
    """Get list of all processed documents"""

    documents = []
    for doc_id, doc_data in processor.processed_documents.items():
        documents.append({
            'id': doc_id,
            'filename': doc_data['original_filename'],
            'file_type': doc_data['file_type'],
            'processed_at': doc_data['processed_at'],
            'confidence_score': doc_data['analysis']['confidence_score'],
            'intelligence_summary': doc_data['analysis']['intelligence_summary']
        })

    return {'documents': documents, 'total_count': len(documents)}


@app.get("/document/{doc_id}")
async def get_document_analysis(doc_id: str):
    """Get detailed analysis for a specific document"""

    if doc_id not in processor.processed_documents:
        raise HTTPException(status_code=404, detail="Document not found")

    return processor.processed_documents[doc_id]


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Intelligence Document Analyzer API", "version": "2.0.0"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)