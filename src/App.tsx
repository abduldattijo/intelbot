// src/App.tsx - Main Application Component (Final Complete Version)

import React, { useState } from 'react';
import './App.css';

// Component imports
import IntelligenceAnalyzer from './components/IntelligenceAnalyzer';
import QueryInterface, { QueryResponse as QueryResponseType } from './components/QueryInterface';
import AnalysisResults from './components/AnalysisResults';
import ForecastingDashboard from './components/ForecastingDashboard';
import GeospatialMap from './components/GeospatialMap';
import DocumentList from './components/DocumentList';
import { FileText, Search, BarChart3, Map, List, Shield, Brain, AlertTriangle } from 'lucide-react';

// Interfaces
export interface Document {
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

type TabType = 'analyzer' | 'query' | 'results' | 'forecasting' | 'map' | 'documents';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('analyzer');
  const [analysisData, setAnalysisData] = useState<Document | null>(null);
  const [queryResponse, setQueryResponse] = useState<QueryResponseType | null>(null);

  const handleDocumentAnalyzed = (document: Document) => {
    setAnalysisData(document);
    setActiveTab('results'); // Automatically switch to results tab
  };

  // UPDATED FUNCTION
  const handleDocumentSelect = async (documentId: string) => {
    try {
      // Fetch the full details for the selected document
      const response = await fetch(`http://localhost:8000/document/${documentId}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch document details. Status: ${response.status}`);
      }
      const documentData: Document = await response.json();

      // Set the analysis data and switch tabs
      setAnalysisData(documentData);
      setActiveTab('results');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
      console.error(errorMessage);
      alert(`Error: ${errorMessage}`);
    }
  };

  const handleQueryResponse = (response: QueryResponseType) => {
    setQueryResponse(response);
  };

  const tabs = [
    { id: 'analyzer' as TabType, label: 'Intelligence Analyzer', icon: FileText, description: 'Upload and analyze documents' },
    { id: 'query' as TabType, label: 'Query Interface', icon: Search, description: 'AI-powered document queries' },
    { id: 'results' as TabType, label: 'Analysis Results', icon: Brain, description: 'View analysis results and insights' },
    { id: 'forecasting' as TabType, label: 'Forecasting', icon: BarChart3, description: 'AI predictions and trends' },
    { id: 'map' as TabType, label: 'Geospatial Map', icon: Map, description: 'Geographic intelligence visualization' },
    { id: 'documents' as TabType, label: 'Document Library', icon: List, description: 'Manage processed documents' }
  ];

  const getTabStatusIndicator = (tabId: TabType) => {
    switch (tabId) {
      case 'results':
        if (analysisData || queryResponse) { return <div className="w-2 h-2 bg-green-500 rounded-full"></div>; }
        break;
      case 'map':
        if (analysisData?.analysis?.geographic_intelligence?.total_locations && analysisData.analysis.geographic_intelligence.total_locations > 0) {
          return <div className="w-2 h-2 bg-blue-500 rounded-full"></div>;
        }
        break;
      case 'query':
        if (queryResponse) { return <div className="w-2 h-2 bg-purple-500 rounded-full"></div>; }
        break;
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <Shield className="h-8 w-8 text-blue-600 mr-3" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Intelligence Document Analyzer</h1>
                <p className="text-sm text-gray-600">AI-Powered Security Intelligence Platform</p>
              </div>
            </div>
            <div className="flex items-center space-x-6">
              {analysisData && (
                <div className="flex items-center space-x-2 text-sm text-green-600">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span>Document Analyzed</span>
                </div>
              )}
              <div className="flex items-center space-x-2 text-sm text-gray-500">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>System Online</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8 overflow-x-auto">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              const statusIndicator = getTabStatusIndicator(tab.id);
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap transition-colors ${isActive ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`}
                  title={tab.description}
                >
                  <Icon className="h-5 w-5" />
                  <span>{tab.label}</span>
                  {statusIndicator}
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto">
        {activeTab === 'analyzer' && <IntelligenceAnalyzer onDocumentAnalyzed={handleDocumentAnalyzed} />}
        {activeTab === 'query' && <QueryInterface onQueryResponse={handleQueryResponse} />}
        {activeTab === 'results' && <AnalysisResults analysisData={analysisData} queryResponse={queryResponse?.response || ''} />}
        {activeTab === 'forecasting' && <ForecastingDashboard />}
        {activeTab === 'map' && <GeospatialMap analysisData={analysisData} />}
        {activeTab === 'documents' && <DocumentList onDocumentSelect={handleDocumentSelect} />}
      </main>

      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex justify-between items-center">
            <p className="text-sm text-gray-500">© 2025 Intelligence Document Analyzer. Advanced AI-powered security intelligence platform.</p>
            <div className="flex items-center space-x-4">
              <div className="text-xs text-gray-400">Version 3.0.0</div>
              {analysisData && <div className="text-xs text-gray-400">Last Analysis: {new Date(analysisData.metadata.uploaded_at).toLocaleString()}</div>}
            </div>
          </div>
        </div>
      </footer>

      {!analysisData && activeTab !== 'analyzer' && (
        <button onClick={() => setActiveTab('analyzer')} className="fixed bottom-6 right-6 bg-blue-600 hover:bg-blue-700 text-white p-4 rounded-full shadow-lg transition-colors z-50" title="Start New Analysis">
          <FileText className="h-6 w-6" />
        </button>
      )}

      {!analysisData && (activeTab === 'results' || activeTab === 'map') && (
        <div className="fixed top-20 right-6 bg-yellow-50 border border-yellow-200 rounded-lg p-4 shadow-lg z-40 max-w-sm">
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 text-yellow-600 mr-2" />
            <div>
              <h3 className="text-sm font-medium text-yellow-800">No Analysis Data</h3>
              <p className="text-xs text-yellow-700 mt-1">Upload a document first to see {activeTab === 'results' ? 'analysis results' : 'geographic intelligence'}.</p>
              <button onClick={() => setActiveTab('analyzer')} className="text-xs text-yellow-800 underline mt-1 hover:text-yellow-900">Go to Analyzer</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;