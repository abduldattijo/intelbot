// src/components/IntelligenceAnalyzer.tsx - Enhanced Multi-Crime Document Analyzer

import React, { useState, useRef, useCallback } from 'react';
import { Upload, FileText, AlertTriangle, CheckCircle, Loader2, Brain, Target, Shield, Database } from 'lucide-react';
import { IntelligenceDocument } from '../App';

interface IntelligenceAnalyzerProps {
  onDocumentAnalyzed: (document: IntelligenceDocument) => void;
}

const IntelligenceAnalyzer: React.FC<IntelligenceAnalyzerProps> = ({ onDocumentAnalyzed }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [analysisStage, setAnalysisStage] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const supportedFormats = [
    {
      name: 'Nigerian Police Reports',
      formats: ['PDF', 'DOCX', 'TXT'],
      examples: [
        'RETURNS ON ARMED BANDITRY/ROBBERY FOR...',
        'RETURNS ON KIDNAP INCIDENTS FOR...',
        'RETURNS ON CULT ACTIVITIES FOR...',
        'RETURNS ON BOKO HARAM ACTIVITIES FOR...',
        'RETURNS ON FARMERS/HERDERS CLASHES FOR...'
      ]
    }
  ];

  const handleFileSelect = useCallback((file: File) => {
    const allowedTypes = [
      'application/pdf',
      'text/plain',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ];

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
    setAnalysisStage('Uploading document...');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Simulate upload progress with more detailed stages
      const stages = [
        'Uploading document...',
        'Extracting text content...',
        'Detecting crime type patterns...',
        'Processing monthly sections...',
        'Extracting incident data...',
        'Analyzing entities and locations...',
        'Generating intelligence summary...',
        'Finalizing analysis...'
      ];

      let currentStage = 0;
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          const newProgress = Math.min(prev + Math.random() * 15, 90);

          // Update stage based on progress
          const stageIndex = Math.floor((newProgress / 90) * stages.length);
          if (stageIndex !== currentStage && stageIndex < stages.length) {
            currentStage = stageIndex;
            setAnalysisStage(stages[stageIndex]);
          }

          return newProgress;
        });
      }, 300);

      const response = await fetch('http://localhost:8000/upload-document', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);
      setAnalysisStage('Analysis complete!');

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
      }

      const data: IntelligenceDocument = await response.json();

      // Brief delay to show completion
      setTimeout(() => {
        onDocumentAnalyzed(data);
        setSelectedFile(null);
        setAnalysisStage('');
      }, 1000);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload and analyze document');
      setAnalysisStage('');
    } finally {
      setUploading(false);
      setTimeout(() => setUploadProgress(0), 2000);
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
          Multi-Crime Intelligence Document Analyzer
        </h1>
        <p className="text-gray-600 max-w-3xl mx-auto">
          Upload Nigerian security intelligence documents for AI-powered analysis supporting all crime types:
          Armed Banditry, Kidnapping, Cult Activities, Boko Haram, Farmers/Herders Clashes, and more.
        </p>
      </div>

      {/* Supported Formats Info */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-blue-900 mb-4 flex items-center">
          <Target className="h-5 w-5 text-blue-600 mr-2" />
          Supported Document Types
        </h3>
        {supportedFormats.map((format, index) => (
          <div key={index} className="mb-4">
            <div className="flex items-center mb-2">
              <Shield className="h-4 w-4 text-blue-600 mr-2" />
              <span className="font-medium text-blue-800">{format.name}</span>
              <span className="ml-2 text-sm text-blue-600">({format.formats.join(', ')})</span>
            </div>
            <div className="ml-6 space-y-1">
              {format.examples.map((example, idx) => (
                <div key={idx} className="text-sm text-blue-700 bg-blue-100 px-2 py-1 rounded inline-block mr-2 mb-1">
                  {example}
                </div>
              ))}
            </div>
          </div>
        ))}
        <div className="mt-4 p-3 bg-blue-100 rounded border border-blue-300">
          <p className="text-sm text-blue-800">
            <strong>✨ New:</strong> The system automatically detects crime types from document headers and processes data separately for each crime category.
          </p>
        </div>
      </div>

      {/* Upload Area */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div
          className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors duration-200 ${
            dragActive 
              ? 'border-blue-500 bg-blue-50' 
              : selectedFile 
                ? 'border-green-500 bg-green-50' 
                : 'border-gray-300 hover:border-gray-400'
          }`}
          onDrop={handleDrop}
          onDragOver={handleDragEvents}
          onDragEnter={handleDragEvents}
          onDragLeave={handleDragEvents}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
            accept=".pdf,.docx,.txt"
          />

          <div className="space-y-4">
            {uploading ? (
              <>
                <Loader2 className="h-12 w-12 text-blue-600 mx-auto animate-spin" />
                <p className="text-lg font-medium text-blue-600">Analyzing Multi-Crime Document...</p>
                <p className="text-sm text-blue-500">{analysisStage}</p>
                <div className="w-full bg-gray-200 rounded-full h-3 max-w-xs mx-auto">
                  <div
                    className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
                <p className="text-xs text-gray-500">{uploadProgress.toFixed(0)}% Complete</p>
              </>
            ) : selectedFile ? (
              <>
                <CheckCircle className="h-12 w-12 text-green-600 mx-auto" />
                <p className="text-lg font-medium text-green-600">{selectedFile.name}</p>
                <p className="text-sm text-gray-500">{formatFileSize(selectedFile.size)} • Ready for multi-crime analysis</p>

                {/* File Preview Info */}
                <div className="mt-4 p-3 bg-green-100 border border-green-200 rounded-lg text-left max-w-md mx-auto">
                  <h4 className="text-sm font-medium text-green-800 mb-2">Analysis Preview:</h4>
                  <ul className="text-xs text-green-700 space-y-1">
                    <li>• Auto-detect crime type from document header</li>
                    <li>• Extract monthly incident statistics</li>
                    <li>• Identify entities, locations, and patterns</li>
                    <li>• Generate intelligent insights and summaries</li>
                    <li>• Store data separately by crime category</li>
                  </ul>
                </div>
              </>
            ) : (
              <>
                <Upload className="h-12 w-12 text-gray-400 mx-auto" />
                <p className="text-lg font-medium text-gray-900">Drop your Nigerian security document here or click to browse</p>
                <div className="text-sm text-gray-500 space-y-1">
                  <p>Supports: PDF, DOCX, TXT • Max size: 50MB</p>
                  <p className="font-medium text-blue-600">All Nigerian crime report formats supported</p>
                </div>
              </>
            )}
          </div>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start">
            <AlertTriangle className="h-5 w-5 text-red-600 mr-2 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-red-800 font-medium">Upload Error</p>
              <p className="text-red-700 text-sm mt-1">{error}</p>
            </div>
          </div>
        )}

        <div className="mt-6">
          <button
            onClick={uploadAndAnalyze}
            disabled={!selectedFile || uploading}
            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-400 disabled:to-gray-400 text-white font-medium py-3 rounded-lg flex items-center justify-center transition-colors"
          >
            {uploading ? (
              <Loader2 className="h-5 w-5 mr-2 animate-spin" />
            ) : (
              <Brain className="h-5 w-5 mr-2" />
            )}
            {uploading ? 'Analyzing...' : 'Start Multi-Crime AI Analysis'}
          </button>
        </div>
      </div>

      {/* Processing Features */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Database className="h-5 w-5 text-purple-600 mr-2" />
          Enhanced Multi-Crime Processing Features
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-3">
              <Target className="h-6 w-6 text-blue-600" />
            </div>
            <h4 className="font-medium text-gray-900 mb-2">Auto Crime Detection</h4>
            <p className="text-sm text-gray-600">Automatically identifies crime type from document headers and categorizes data accordingly.</p>
          </div>

          <div className="text-center">
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-3">
              <Database className="h-6 w-6 text-green-600" />
            </div>
            <h4 className="font-medium text-gray-900 mb-2">Separated Data Storage</h4>
            <p className="text-sm text-gray-600">Stores incident data separately by crime type for accurate comparisons and forecasting.</p>
          </div>

          <div className="text-center">
            <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-3">
              <Brain className="h-6 w-6 text-purple-600" />
            </div>
            <h4 className="font-medium text-gray-900 mb-2">Enhanced Analytics</h4>
            <p className="text-sm text-gray-600">Generate crime-specific insights, trends, and predictions using advanced AI models.</p>
          </div>

          <div className="text-center">
            <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center mx-auto mb-3">
              <AlertTriangle className="h-6 w-6 text-red-600" />
            </div>
            <h4 className="font-medium text-gray-900 mb-2">Threat Assessment</h4>
            <p className="text-sm text-gray-600">Automated threat level classification and risk pattern identification for each crime type.</p>
          </div>

          <div className="text-center">
            <div className="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center mx-auto mb-3">
              <Shield className="h-6 w-6 text-yellow-600" />
            </div>
            <h4 className="font-medium text-gray-900 mb-2">Geographic Mapping</h4>
            <p className="text-sm text-gray-600">Extract and map Nigerian state-level incident data with threat visualization.</p>
          </div>

          <div className="text-center">
            <div className="w-12 h-12 bg-indigo-100 rounded-lg flex items-center justify-center mx-auto mb-3">
              <FileText className="h-6 w-6 text-indigo-600" />
            </div>
            <h4 className="font-medium text-gray-900 mb-2">Table Preservation</h4>
            <p className="text-sm text-gray-600">Maintains table structure and formatting for accurate data extraction and analysis.</p>
          </div>
        </div>
      </div>

      {/* Success Tips */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Tips for Best Results</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-700">
          <div>
            <h4 className="font-medium text-gray-900 mb-2">Document Format:</h4>
            <ul className="space-y-1">
              <li>• Use clear, readable document scans</li>
              <li>• Ensure proper document structure</li>
              <li>• Include complete monthly headers</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-gray-900 mb-2">Crime Type Headers:</h4>
            <ul className="space-y-1">
              <li>• "RETURNS ON [CRIME TYPE] FOR [MONTH], [YEAR]"</li>
              <li>• Clear monthly section divisions</li>
              <li>• Consistent numerical formatting</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default IntelligenceAnalyzer;