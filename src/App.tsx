// src/App.tsx - FINAL AND COMPLETE

import React, { useState } from 'react';
import './App.css';

// Component imports
import IntelligenceAnalyzer from './components/IntelligenceAnalyzer';
import QueryInterface, { QueryResponse as QueryResponseType } from './components/QueryInterface';
import AnalysisResults from './components/AnalysisResults';
import ForecastingDashboard from './components/ForecastingDashboard';
import GeospatialMap from './components/GeospatialMap';
import DocumentList from './components/DocumentList';
import ComparisonDashboard from './components/ComparisonDashboard';
import { FileText, Search, BarChart3, Map, List, Brain, BarChart } from 'lucide-react';


export interface IntelligenceDocument {
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
      dates: string[];
      weapons: string[];
      vehicles: string[];
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
      // <<< FIX: Added 'threat_level' to the coordinate object definition >>>
      coordinates: Array<{
        latitude: number;
        longitude: number;
        location_name: string;
        confidence: number;
        threat_level: 'low' | 'medium' | 'high';
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
      crime_trends: Array<any>;
    };
    relationships: Array<any>;
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

type TabType = 'analyzer' | 'query' | 'results' | 'forecasting' | 'map' | 'documents' | 'comparison';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('analyzer');
  const [analysisData, setAnalysisData] = useState<IntelligenceDocument | null>(null);
  const [queryResponse, setQueryResponse] = useState<QueryResponseType | null>(null);

  const handleDocumentAnalyzed = (document: IntelligenceDocument) => {
    setAnalysisData(document);
    setActiveTab('results');
  };

  const handleDocumentSelect = async (documentId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/document/${documentId}`);
      if (!response.ok) throw new Error(`Failed to fetch document details. Status: ${response.status}`);
      const documentData: IntelligenceDocument = await response.json();
      setAnalysisData(documentData);
      setActiveTab('results');
    } catch (error) {
      alert(`Error: ${error instanceof Error ? error.message : 'An unknown error occurred'}`);
    }
  };

  const handleQueryResponse = (response: QueryResponseType) => {
    setQueryResponse(response);
  };

  const tabs = [
    { id: 'analyzer' as TabType, label: 'Intelligence Analyzer', icon: FileText, description: 'Upload and analyze documents' },
    { id: 'documents' as TabType, label: 'Document Library', icon: List, description: 'Manage processed documents' },
    { id: 'query' as TabType, label: 'Query Interface', icon: Search, description: 'AI-powered document queries' },
    { id: 'results' as TabType, label: 'Analysis Results', icon: Brain, description: 'View analysis results and insights' },
    { id: 'comparison' as TabType, label: 'Monthly Comparison', icon: BarChart, description: 'Compare two months of data' },
    { id: 'forecasting' as TabType, label: 'Forecasting', icon: BarChart3, description: 'AI predictions and trends' },
    { id: 'map' as TabType, label: 'Geospatial Map', icon: Map, description: 'Geographic intelligence visualization' },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-blue-600 mr-3" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Intelligence Document Analyzer</h1>
                <p className="text-sm text-gray-600">AI-Powered Security Intelligence Platform</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm text-gray-500">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>System Online</span>
            </div>
          </div>
        </div>
      </header>

      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8 overflow-x-auto">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap transition-colors ${activeTab === tab.id ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`}
                  title={tab.description}
                >
                  <>
                    <Icon className="h-5 w-5" />
                    <span>{tab.label}</span>
                  </>
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {activeTab === 'analyzer' && <IntelligenceAnalyzer onDocumentAnalyzed={handleDocumentAnalyzed} />}
        {activeTab === 'documents' && <DocumentList onDocumentSelect={handleDocumentSelect} />}
        {activeTab === 'query' && <QueryInterface onQueryResponse={handleQueryResponse} />}
        {activeTab === 'results' && <AnalysisResults analysisData={analysisData} queryResponse={queryResponse?.response || ''} />}
        {activeTab === 'comparison' && <ComparisonDashboard />}
        {activeTab === 'forecasting' && <ForecastingDashboard />}
        {activeTab === 'map' && <GeospatialMap analysisData={analysisData} />}
      </main>

      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-500">© 2025 Intelligence Document Analyzer Platform</p>
        </div>
      </footer>
    </div>
  );
};

export default App;