// src/components/IntelligenceAnalyzer.tsx - FINAL AND CORRECTED

import React, { useState, useRef, useCallback } from 'react';
import { Upload, FileText, AlertTriangle, CheckCircle, Loader2, Brain } from 'lucide-react';
import { IntelligenceDocument } from '../App'; // <<< FIX: Import the canonical interface from App.tsx

// <<< FIX: The local 'Document' interface has been REMOVED >>>

interface IntelligenceAnalyzerProps {
  // <<< FIX: Use the imported interface for the prop >>>
  onDocumentAnalyzed: (document: IntelligenceDocument) => void;
}

const IntelligenceAnalyzer: React.FC<IntelligenceAnalyzerProps> = ({ onDocumentAnalyzed }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback((file: File) => {
    const allowedTypes = ['application/pdf', 'text/plain', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
    if (!allowedTypes.includes(file.type)) {
      setError('Unsupported file type. Please upload PDF, DOCX, or TXT files.');
      return;
    }
    if (file.size > 50 * 1024 * 1024) { // 50MB limit
      setError('File size too large. Please upload files smaller than 50MB.');
      return;
    }
    setSelectedFile(file);
    setError(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  }, [handleFileSelect]);

  const handleDragEvents = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const uploadAndAnalyze = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setUploadProgress(0);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + Math.random() * 20, 90));
      }, 200);

      const response = await fetch('http://localhost:8000/upload-document', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
      }

      const data: IntelligenceDocument = await response.json();
      onDocumentAnalyzed(data);
      setSelectedFile(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload and analyze document');
    } finally {
      setUploading(false);
      setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
  };

  return (
    <div className="p-6 space-y-6">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2 flex items-center justify-center">
          <FileText className="h-8 w-8 text-blue-600 mr-3" />
          Intelligence Document Analyzer
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Upload intelligence documents for AI-powered analysis, entity extraction, and geospatial threat mapping.
        </p>
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div
          className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors duration-200 ${dragActive ? 'border-blue-500 bg-blue-50' : selectedFile ? 'border-green-500 bg-green-50' : 'border-gray-300 hover:border-gray-400'}`}
          onDrop={handleDrop}
          onDragOver={handleDragEvents}
          onDragEnter={handleDragEvents}
          onDragLeave={handleDragEvents}
          onClick={() => fileInputRef.current?.click()}
        >
          <input ref={fileInputRef} type="file" className="hidden" onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])} />
          <div className="space-y-4">
            {uploading ? (
              <>
                <Loader2 className="h-12 w-12 text-blue-600 mx-auto animate-spin" />
                <p className="text-lg font-medium text-blue-600">Analyzing Document...</p>
                <div className="w-full bg-gray-200 rounded-full h-2 max-w-xs mx-auto"><div className="bg-blue-600 h-2 rounded-full" style={{ width: `${uploadProgress}%` }}></div></div>
              </>
            ) : selectedFile ? (
              <>
                <CheckCircle className="h-12 w-12 text-green-600 mx-auto" />
                <p className="text-lg font-medium text-green-600">{selectedFile.name}</p>
                <p className="text-sm text-gray-500">{formatFileSize(selectedFile.size)} • Ready for analysis</p>
              </>
            ) : (
              <>
                <Upload className="h-12 w-12 text-gray-400 mx-auto" />
                <p className="text-lg font-medium text-gray-900">Drop your document here or click to browse</p>
                <div className="text-xs text-gray-400">Supports: PDF, DOCX, TXT</div>
              </>
            )}
          </div>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center">
            <AlertTriangle className="h-5 w-5 text-red-600 mr-2" />
            <p className="text-red-800 font-medium">{error}</p>
          </div>
        )}

        <div className="mt-6">
          <button onClick={uploadAndAnalyze} disabled={!selectedFile || uploading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-3 rounded-lg flex items-center justify-center">
            {uploading ? <Loader2 className="h-5 w-5 mr-2 animate-spin" /> : <Brain className="h-5 w-5 mr-2" />}
            {uploading ? 'Analyzing...' : 'Start AI Analysis'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default IntelligenceAnalyzer;