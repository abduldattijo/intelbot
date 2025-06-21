// src/components/AnalysisResults.tsx - Enhanced Analysis Results Component

import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import { FileText, MapPin, Calendar, Users, Shield, AlertTriangle, Brain, TrendingUp, Eye, Download } from 'lucide-react';
import { Document } from '../App';

interface AnalysisResultsProps {
  analysisData: Document | null;
  queryResponse: string;
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ analysisData, queryResponse }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'detailed' | 'query'>('overview');

  if (!analysisData && !queryResponse) {
    return (
      <div className="p-8 text-center">
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-8">
          <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Analysis Data</h3>
          <p className="text-gray-600">
            Upload a document or submit a query to view analysis results here.
          </p>
        </div>
      </div>
    );
  }

  const formatResponse = (response: string) => {
    const sections = response.split('\n\n');
    return sections.map((section, index) => {
      if (section.startsWith('**') && section.endsWith('**')) {
        return (
          <h4 key={index} className="font-semibold text-gray-900 mt-4 mb-2">
            {section.replace(/\*\*/g, '')}
          </h4>
        );
      } else if (section.includes('•')) {
        const items = section.split('\n').filter(item => item.trim());
        return (
          <ul key={index} className="list-disc list-inside space-y-1 mb-3">
            {items.map((item, itemIndex) => (
              <li key={itemIndex} className="text-gray-700">
                {item.replace('•', '').trim()}
              </li>
            ))}
          </ul>
        );
      } else {
        return (
          <p key={index} className="text-gray-700 mb-3 leading-relaxed">
            {section}
          </p>
        );
      }
    });
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

  const getThreatColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'high': return 'text-red-600 bg-red-50 border-red-200';
      case 'medium': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'low': return 'text-green-600 bg-green-50 border-green-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899'];

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Eye },
    { id: 'detailed', label: 'Detailed Analysis', icon: Brain },
    { id: 'query', label: 'Query Results', icon: FileText }
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Analysis Results</h2>
          <p className="text-gray-600 mt-1">
            {analysisData ? `Document: ${analysisData.metadata.filename}` : 'Query Response Analysis'}
          </p>
        </div>

        {/* Export Button */}
        <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
          <Download className="h-4 w-4" />
          <span>Export Report</span>
        </button>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            const isDisabled = (tab.id === 'detailed' && !analysisData) || (tab.id === 'query' && !queryResponse);

            return (
              <button
                key={tab.id}
                onClick={() => !isDisabled && setActiveTab(tab.id as any)}
                disabled={isDisabled}
                className={`
                  flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm transition-colors
                  ${isActive 
                    ? 'border-blue-500 text-blue-600' 
                    : isDisabled 
                      ? 'border-transparent text-gray-400 cursor-not-allowed'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }
                `}
              >
                <Icon className="h-4 w-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && analysisData && (
        <div className="space-y-6">
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="bg-white p-6 rounded-lg shadow border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Confidence Score</p>
                  <p className="text-2xl font-bold text-blue-600">
                    {(analysisData.analysis.confidence_score * 100).toFixed(1)}%
                  </p>
                </div>
                <Shield className="h-8 w-8 text-blue-500" />
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg shadow border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Threat Level</p>
                  <p className={`text-xl font-bold ${
                    analysisData.analysis.sentiment_analysis.threat_level === 'High' ? 'text-red-600' :
                    analysisData.analysis.sentiment_analysis.threat_level === 'Medium' ? 'text-orange-600' : 'text-green-600'
                  }`}>
                    {analysisData.analysis.sentiment_analysis.threat_level}
                  </p>
                </div>
                <AlertTriangle className={`h-8 w-8 ${
                  analysisData.analysis.sentiment_analysis.threat_level === 'High' ? 'text-red-500' :
                  analysisData.analysis.sentiment_analysis.threat_level === 'Medium' ? 'text-orange-500' : 'text-green-500'
                }`} />
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg shadow border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Locations</p>
                  <p className="text-2xl font-bold text-purple-600">
                    {analysisData.analysis.geographic_intelligence.total_locations}
                  </p>
                </div>
                <MapPin className="h-8 w-8 text-purple-500" />
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg shadow border">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">Word Count</p>
                  <p className="text-2xl font-bold text-gray-600">
                    {analysisData.analysis.text_statistics.word_count.toLocaleString()}
                  </p>
                </div>
                <FileText className="h-8 w-8 text-gray-500" />
              </div>
            </div>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Numerical Intelligence Chart */}
            {generateChartData().length > 0 && (
              <div className="bg-white p-6 rounded-lg shadow border">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Intelligence Metrics</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={generateChartData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Crime Patterns Chart */}
            {generateCrimeData().length > 0 && (
              <div className="bg-white p-6 rounded-lg shadow border">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Crime Patterns</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={generateCrimeData()}
                        cx="50%"
                        cy="50%"
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
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>

          {/* Intelligence Summary */}
          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Brain className="h-5 w-5 text-blue-600 mr-2" />
              AI Intelligence Summary
            </h3>
            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
              <p className="text-blue-800 leading-relaxed">
                {analysisData.analysis.intelligence_summary}
              </p>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'detailed' && analysisData && (
        <div className="space-y-6">
          {/* Document Classification */}
          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Document Classification</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <span className="text-sm font-medium text-gray-600">Document Type:</span>
                <p className="text-lg font-bold text-blue-600 capitalize">
                  {analysisData.analysis.document_classification.primary_type.replace('_', ' ')}
                </p>
              </div>
              <div>
                <span className="text-sm font-medium text-gray-600">Security Classification:</span>
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

          {/* Geographic Intelligence */}
          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <MapPin className="h-5 w-5 text-red-600 mr-2" />
              Geographic Intelligence
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-gray-900 mb-2">Affected States:</h4>
                {analysisData.analysis.geographic_intelligence.states.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {analysisData.analysis.geographic_intelligence.states.map((state, index) => (
                      <span key={index} className="px-2 py-1 bg-red-100 text-red-700 rounded text-sm">
                        {state}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500">No states identified</p>
                )}
              </div>
              <div>
                <h4 className="font-medium text-gray-900 mb-2">Other Locations:</h4>
                {analysisData.analysis.geographic_intelligence.other_locations.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {analysisData.analysis.geographic_intelligence.other_locations.map((location, index) => (
                      <span key={index} className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-sm">
                        {location}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500">No other locations identified</p>
                )}
              </div>
            </div>
          </div>

          {/* Entities */}
          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Extracted Entities</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(analysisData.analysis.entities).map(([type, entities]) => (
                <div key={type}>
                  <h4 className="font-medium text-gray-900 mb-2 capitalize">{type}:</h4>
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

          {/* Temporal Intelligence */}
          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Calendar className="h-5 w-5 text-orange-600 mr-2" />
              Temporal Intelligence
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-gray-900 mb-2">Months Mentioned:</h4>
                {analysisData.analysis.temporal_intelligence.months_mentioned.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {analysisData.analysis.temporal_intelligence.months_mentioned.map((month, index) => (
                      <span key={index} className="px-2 py-1 bg-orange-100 text-orange-700 rounded text-sm">
                        {month}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500">No months identified</p>
                )}
              </div>
              <div>
                <h4 className="font-medium text-gray-900 mb-2">Years Mentioned:</h4>
                {analysisData.analysis.temporal_intelligence.years_mentioned.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {analysisData.analysis.temporal_intelligence.years_mentioned.map((year, index) => (
                      <span key={index} className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-sm">
                        {year}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500">No years identified</p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'query' && queryResponse && (
        <div className="space-y-6">
          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Brain className="h-5 w-5 text-green-600 mr-2" />
              AI Query Response
            </h3>
            <div className="prose prose-gray max-w-none">
              {formatResponse(queryResponse)}
            </div>
          </div>
        </div>
      )}

      {/* Empty State for Query Tab */}
      {activeTab === 'query' && !queryResponse && (
        <div className="text-center bg-gray-50 border border-gray-200 rounded-lg p-8">
          <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Query Response</h3>
          <p className="text-gray-600">
            Submit a query in the Query Interface to see AI-powered analysis results here.
          </p>
        </div>
      )}
    </div>
  );
};

export default AnalysisResults;