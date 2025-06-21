// src/components/IntelligenceAnalyzer.tsx - Main Intelligence Analysis Component (Fixed)

import React, { useState, useRef, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area } from 'recharts';
import { TrendingUp, MapPin, AlertTriangle, Target, Calendar, Users, Shield, Brain, FileText, Activity, Upload, CheckCircle, Loader2, Globe, Zap } from 'lucide-react';

// Interfaces
interface Document {
  id: string;
  content: string;
  metadata: {
    filename: string;
    file_type: string;
    uploaded_at: string;
    file_size: number;
  };
  analysis: {
    document_classification: {
      primary_type: string;
      sub_types: string[];
      confidence: number;
      security_classification: string;
    };
    entities: {
      persons: string[];
      organizations: string[];
      locations: string[];
      weapons: string[];
      vehicles: string[];
      dates: string[];
    };
    sentiment_analysis: {
      overall_sentiment: string;
      sentiment_score: number;
      threat_level: string;
      urgency_indicators: string[];
    };
    geographic_intelligence: {
      states: string[];
      cities: string[];
      countries: string[];
      coordinates: Array<{
        latitude: number;
        longitude: number;
        location_name: string;
        confidence: number;
      }>;
      total_locations: number;
      other_locations: string[];
    };
    temporal_intelligence: {
      dates_mentioned: string[];
      time_periods: string[];
      months_mentioned: string[];
      years_mentioned: string[];
      temporal_patterns: string[];
    };
    numerical_intelligence: {
      incidents: number[];
      casualties: number[];
      weapons: number[];
      arrests: number[];
      monetary_values: number[];
    };
    crime_patterns: {
      primary_crimes: Array<[string, number]>;
      crime_frequency: Record<string, number>;
      crime_trends: Array<{
        crime_type: string;
        trend: string;
        confidence: number;
      }>;
    };
    relationships: Array<{
      entity1: string;
      entity2: string;
      relationship_type: string;
      confidence: number;
    }>;
    text_statistics: {
      word_count: number;
      sentence_count: number;
      paragraph_count: number;
      readability_score: number;
      language: string;
    };
    intelligence_summary: string;
    confidence_score: number;
    processing_time: number;
  };
}

interface IntelligenceAnalyzerProps {
  onDocumentAnalyzed: (document: Document) => void;
}

const IntelligenceAnalyzer: React.FC<IntelligenceAnalyzerProps> = ({ onDocumentAnalyzed }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [analysisData, setAnalysisData] = useState<Document | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback((file: File) => {
    const allowedTypes = [
      'application/pdf',
      'text/plain',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'text/csv',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    ];

    if (!allowedTypes.includes(file.type)) {
      setError('Unsupported file type. Please upload PDF, DOC, DOCX, TXT, CSV, or Excel files.');
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

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
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
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + Math.random() * 20;
        });
      }, 200);

      const response = await fetch('http://localhost:8000/upload-document', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setAnalysisData(data);
      onDocumentAnalyzed(data);

      // Reset form
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
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
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const generateChartData = () => {
    if (!analysisData) return [];

    const numerical = analysisData.analysis.numerical_intelligence;
    return [
      { name: 'Incidents', value: Math.max(...(numerical.incidents || [0])) },
      { name: 'Casualties', value: Math.max(...(numerical.casualties || [0])) },
      { name: 'Weapons', value: Math.max(...(numerical.weapons || [0])) },
      { name: 'Arrests', value: Math.max(...(numerical.arrests || [0])) }
    ].filter(item => item.value > 0);
  };

  const generateCrimeData = () => {
    if (!analysisData) return [];

    return analysisData.analysis.crime_patterns.primary_crimes.map(([crime, count]) => ({
      name: crime.replace('_', ' ').toUpperCase(),
      value: count
    }));
  };

  const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899'];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2 flex items-center justify-center">
          <Shield className="h-8 w-8 text-blue-600 mr-3" />
          Intelligence Document Analyzer
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Upload intelligence documents for AI-powered analysis including threat assessment,
          entity extraction, geospatial intelligence, and comprehensive security insights.
        </p>
      </div>

      {/* Upload Section */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
          <FileText className="h-5 w-5 text-blue-600 mr-2" />
          Document Upload & Analysis
        </h2>

        {/* Drag and Drop Area */}
        <div
          className={`
            relative border-2 border-dashed rounded-lg p-8 text-center transition-colors duration-200
            ${dragActive 
              ? 'border-blue-500 bg-blue-50' 
              : selectedFile 
                ? 'border-green-500 bg-green-50' 
                : 'border-gray-300 hover:border-gray-400'
            }
          `}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".pdf,.doc,.docx,.txt,.csv,.xls,.xlsx"
            onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
          />

          <div className="space-y-4">
            {uploading ? (
              <>
                <Loader2 className="h-12 w-12 text-blue-600 mx-auto animate-spin" />
                <p className="text-lg font-medium text-blue-600">
                  Analyzing Document...
                </p>
                <div className="w-full bg-gray-200 rounded-full h-2 max-w-xs mx-auto">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
                <p className="text-sm text-gray-500">{uploadProgress.toFixed(0)}% Complete</p>
              </>
            ) : selectedFile ? (
              <>
                <CheckCircle className="h-12 w-12 text-green-600 mx-auto" />
                <div>
                  <p className="text-lg font-medium text-green-600">
                    {selectedFile.name}
                  </p>
                  <p className="text-sm text-gray-500">
                    {formatFileSize(selectedFile.size)} • Ready for analysis
                  </p>
                </div>
              </>
            ) : (
              <>
                <Upload className="h-12 w-12 text-gray-400 mx-auto" />
                <div>
                  <p className="text-lg font-medium text-gray-900">
                    Drop your intelligence document here
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    or click to browse files
                  </p>
                </div>
                <div className="text-xs text-gray-400">
                  Supports: PDF, DOC, DOCX, TXT, CSV, Excel (Max 50MB)
                </div>
              </>
            )}
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center">
              <AlertTriangle className="h-5 w-5 text-red-600 mr-2" />
              <p className="text-red-800 font-medium">Upload Error</p>
            </div>
            <p className="text-red-700 mt-1">{error}</p>
          </div>
        )}

        {/* Upload Button */}
        <div className="mt-6">
          <button
            onClick={uploadAndAnalyze}
            disabled={!selectedFile || uploading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-medium py-3 px-6 rounded-lg transition-colors duration-200 flex items-center justify-center"
          >
            {uploading ? (
              <>
                <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Brain className="h-5 w-5 mr-2" />
                Start AI Analysis
              </>
            )}
          </button>
        </div>
      </div>

      {/* Analysis Results */}
      {analysisData && (
        <div className="space-y-6">
          {/* Document Overview */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
              <FileText className="h-5 w-5 text-green-600 mr-2" />
              Document Analysis Complete
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">
                  {(analysisData.analysis.confidence_score * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-blue-600 font-medium">Confidence Score</div>
              </div>

              <div className="bg-red-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-red-600">
                  {analysisData.analysis.sentiment_analysis.threat_level}
                </div>
                <div className="text-sm text-red-600 font-medium">Threat Level</div>
              </div>

              <div className="bg-purple-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {analysisData.analysis.geographic_intelligence.total_locations}
                </div>
                <div className="text-sm text-purple-600 font-medium">Locations Found</div>
              </div>

              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-gray-600">
                  {analysisData.analysis.text_statistics.word_count.toLocaleString()}
                </div>
                <div className="text-sm text-gray-600 font-medium">Word Count</div>
              </div>
            </div>
          </div>

          {/* Charts Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Numerical Intelligence Chart */}
            {generateChartData().length > 0 && (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <Activity className="h-5 w-5 text-blue-600 mr-2" />
                  Intelligence Metrics
                </h3>
                <div className="h-64">
                  <BarChart width={400} height={250} data={generateChartData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip
                      formatter={(value: any, name: string) => [
                        typeof value === 'number' ? value.toFixed(0) : value,
                        name
                      ]}
                    />
                    <Bar dataKey="value" fill="#3b82f6" />
                  </BarChart>
                </div>
              </div>
            )}

            {/* Crime Patterns Chart */}
            {generateCrimeData().length > 0 && (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <AlertTriangle className="h-5 w-5 text-red-600 mr-2" />
                  Crime Patterns
                </h3>
                <div className="h-64">
                  <PieChart width={400} height={250}>
                    <Pie
                      data={generateCrimeData()}
                      cx={200}
                      cy={125}
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {generateCrimeData().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip
                      formatter={(value: any, name: string) => [
                        typeof value === 'number' ? value.toFixed(0) : value,
                        name
                      ]}
                    />
                  </PieChart>
                </div>
              </div>
            )}
          </div>

          {/* Intelligence Summary */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Brain className="h-5 w-5 text-green-600 mr-2" />
              AI Intelligence Summary
            </h3>
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <p className="text-green-800 leading-relaxed">
                {analysisData.analysis.intelligence_summary}
              </p>
            </div>
          </div>

          {/* Entities Analysis */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Users className="h-5 w-5 text-purple-600 mr-2" />
              Extracted Entities
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(analysisData.analysis.entities).map(([type, entities]) => (
                <div key={type} className="space-y-2">
                  <h4 className="font-medium text-gray-900 capitalize">{type}:</h4>
                  {entities.length > 0 ? (
                    <div className="space-y-1">
                      {entities.slice(0, 5).map((entity, index) => (
                        <span key={index} className="block px-2 py-1 bg-gray-100 text-gray-700 rounded text-sm">
                          {entity}
                        </span>
                      ))}
                      {entities.length > 5 && (
                        <span className="text-xs text-gray-500">+{entities.length - 5} more</span>
                      )}
                    </div>
                  ) : (
                    <p className="text-gray-500 text-sm">None identified</p>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Geographic Intelligence */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <MapPin className="h-5 w-5 text-red-600 mr-2" />
              Geographic Intelligence
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-gray-900 mb-2">States/Regions:</h4>
                <div className="space-y-1">
                  {analysisData.analysis.geographic_intelligence.states.map((state, index) => (
                    <span key={index} className="inline-block px-2 py-1 bg-red-100 text-red-700 rounded text-sm mr-2 mb-1">
                      {state}
                    </span>
                  ))}
                  {analysisData.analysis.geographic_intelligence.states.length === 0 && (
                    <p className="text-gray-500 text-sm">No states identified</p>
                  )}
                </div>
              </div>
              <div>
                <h4 className="font-medium text-gray-900 mb-2">Other Locations:</h4>
                <div className="space-y-1">
                  {analysisData.analysis.geographic_intelligence.other_locations.map((location, index) => (
                    <span key={index} className="inline-block px-2 py-1 bg-blue-100 text-blue-700 rounded text-sm mr-2 mb-1">
                      {location}
                    </span>
                  ))}
                  {analysisData.analysis.geographic_intelligence.other_locations.length === 0 && (
                    <p className="text-gray-500 text-sm">No other locations identified</p>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Geospatial Intelligence Note */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
            <div className="flex items-center">
              <MapPin className="h-5 w-5 text-blue-600 mr-2" />
              <h3 className="text-lg font-medium text-blue-800">Geospatial Intelligence</h3>
            </div>
            <p className="text-blue-700 mt-2">
              Geographic data and location intelligence are available in the dedicated Geospatial Map tab.
              Switch to the Map view to explore interactive intelligence points and threat analysis.
            </p>
          </div>

          {/* Document Classification */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Shield className="h-5 w-5 text-blue-600 mr-2" />
              Document Classification
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <span className="text-sm font-medium text-gray-600">Primary Type:</span>
                <p className="text-lg font-bold text-blue-600 capitalize">
                  {analysisData.analysis.document_classification.primary_type.replace('_', ' ')}
                </p>
              </div>
              <div>
                <span className="text-sm font-medium text-gray-600">Security Level:</span>
                <p className="text-lg font-bold text-red-600">
                  {analysisData.analysis.document_classification.security_classification}
                </p>
              </div>
              <div>
                <span className="text-sm font-medium text-gray-600">Classification Confidence:</span>
                <p className="text-lg font-bold text-green-600">
                  {(analysisData.analysis.document_classification.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>

          {/* Processing Information */}
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between text-sm text-gray-600">
              <span>
                Analysis completed in {analysisData.analysis.processing_time.toFixed(2)} seconds
              </span>
              <span>
                Document ID: {analysisData.id}
              </span>
              <span>
                Processed: {new Date(analysisData.metadata.uploaded_at).toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default IntelligenceAnalyzer;