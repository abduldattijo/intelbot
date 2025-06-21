// src/App.tsx - Enhanced Main Application Component

import React, { useState, useCallback } from 'react';
import './App.css';
import DocumentUpload from './components/DocumentUpload';
import QueryInterface from './components/QueryInterface';
import AnalysisResults from './components/AnalysisResults';
import ForecastingDashboard from './components/ForecastingDashboard';
import GeospatialMap from './components/GeospatialMap';
import DocumentList from './components/DocumentList';
import { FileText, Search, BarChart3, Map, List, Shield, Brain, AlertTriangle } from 'lucide-react';

export interface Document {
  document_id: string;
  status: string;
  analysis: {
    document_classification: {
      primary_type: string;
      confidence: number;
      security_classification: string;
    };
    text_statistics: {
      word_count: number;
      sentence_count: number;
      character_count: number;
    };
    numerical_intelligence: {
      incidents: number[];
      casualties: number[];
      weapons: number[];
      arrests: number[];
    };
    geographic_intelligence: {
      states: string[];
      zones: string[];
      other_locations: string[];
      coordinates: Record<string, { latitude: number; longitude: number }>;
      total_locations: number;
    };
    temporal_intelligence: {
      months_mentioned: string[];
      years_mentioned: string[];
      time_periods: string[];
      dates_found: string[][];
      temporal_density: number;
    };
    sentiment_analysis: {
      polarity: number;
      subjectivity: number;
      threat_level: string;
      threat_indicators: number;
    };
    entities: {
      persons: string[];
      organizations: string[];
      locations: string[];
      dates: string[];
      money: string[];
      weapons: string[];
    };
    crime_patterns: {
      crime_frequencies: Record<string, number>;
      primary_crimes: [string, number][];
      modus_operandi_indicators: string[];
    };
    intelligence_summary: string;
    confidence_score: number;
  };
  metadata: {
    filename: string;
    file_type: string;
    text_length: number;
    processed_at: string;
  };
}

export interface QueryResponse {
  response: string;
}

export interface ForecastData {
  historical_data: {
    months: string[];
    incidents: number[];
  };
  forecasts: {
    months: string[];
    predictions: number[];
    confidence_intervals: Array<{
      lower: number;
      upper: number;
      prediction: number;
    }>;
  };
  model_metrics: {
    r_squared: number;
    mean_squared_error: number;
    trend_coefficient: number;
  };
  insights: string[];
}

export interface GeospatialData {
  incidents: Array<{
    location: string;
    latitude: number;
    longitude: number;
    incidents: number;
    casualties: number;
    document_id: string;
    threat_level: string;
    primary_crimes: string[];
    document_summary: string;
  }>;
  hotspots: Record<string, {
    total_incidents: number;
    total_casualties: number;
    threat_level: string;
    coordinates: { latitude: number; longitude: number };
  }>;
  total_locations: number;
}

type ActiveTab = 'upload' | 'query' | 'analysis' | 'forecasting' | 'mapping' | 'documents';

function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('upload');
  const [analysisData, setAnalysisData] = useState<Document | null>(null);
  const [queryResponse, setQueryResponse] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);

  const handleDocumentProcessed = useCallback((data: Document) => {
    setAnalysisData(data);
    setActiveTab('analysis');
  }, []);

  const handleQueryResponse = useCallback((response: string) => {
    setQueryResponse(response);
  }, []);

  const tabs = [
    { id: 'upload' as const, label: 'Document Upload', icon: FileText, color: 'bg-blue-500' },
    { id: 'query' as const, label: 'Query Interface', icon: Search, color: 'bg-green-500' },
    { id: 'analysis' as const, label: 'Analysis Results', icon: Brain, color: 'bg-purple-500' },
    { id: 'forecasting' as const, label: 'Forecasting', icon: BarChart3, color: 'bg-orange-500' },
    { id: 'mapping' as const, label: 'Geospatial Map', icon: Map, color: 'bg-red-500' },
    { id: 'documents' as const, label: 'Document List', icon: List, color: 'bg-gray-500' }
  ];

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'upload':
        return (
          <DocumentUpload
            onDocumentProcessed={handleDocumentProcessed}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
          />
        );
      case 'query':
        return (
          <QueryInterface
            onQueryResponse={handleQueryResponse}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
          />
        );
      case 'analysis':
        return (
          <AnalysisResults
            analysisData={analysisData}
            queryResponse={queryResponse}
          />
        );
      case 'forecasting':
        return <ForecastingDashboard />;
      case 'mapping':
        return <GeospatialMap />;
      case 'documents':
        return <DocumentList onDocumentSelect={setAnalysisData} />;
      default:
        return <DocumentUpload onDocumentProcessed={handleDocumentProcessed} isLoading={isLoading} setIsLoading={setIsLoading} />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Shield className="h-8 w-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">Intelligence Document Analyzer</h1>
                <p className="text-sm text-gray-600">Enhanced AI-Powered Security Intelligence Platform v3.0</p>
              </div>
            </div>

            {/* Status Indicator */}
            <div className="flex items-center space-x-2">
              <div className="flex items-center space-x-2 px-3 py-1 bg-green-100 rounded-full">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm font-medium text-green-700">System Online</span>
              </div>
              {isLoading && (
                <div className="flex items-center space-x-2 px-3 py-1 bg-blue-100 rounded-full">
                  <AlertTriangle className="w-4 h-4 text-blue-600 animate-spin" />
                  <span className="text-sm font-medium text-blue-700">Processing</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;

              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`
                    flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors duration-200
                    ${isActive 
                      ? 'border-blue-500 text-blue-600' 
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }
                  `}
                >
                  <div className={`p-1 rounded ${isActive ? tab.color : 'bg-gray-200'}`}>
                    <Icon className={`h-4 w-4 ${isActive ? 'text-white' : 'text-gray-500'}`} />
                  </div>
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
          {renderActiveTab()}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-8 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <div>
              <p className="text-sm text-gray-300">
                © 2024 Intelligence Document Analyzer. Enhanced AI-Powered Security Intelligence Platform.
              </p>
              <p className="text-xs text-gray-400 mt-1">
                Featuring ML forecasting, geospatial mapping, and advanced NLP analysis.
              </p>
            </div>
            <div className="text-xs text-gray-400">
              Version 3.0.0 | Last Updated: {new Date().toLocaleDateString()}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;