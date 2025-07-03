// src/components/AnalysisResults.tsx - FINAL AND COMPLETE

import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { FileText, MapPin, Calendar, Users, Shield, AlertTriangle, Brain, Eye, Download } from 'lucide-react';
import { IntelligenceDocument } from '../App';

interface AnalysisResultsProps {
  analysisData: IntelligenceDocument | null;
  queryResponse: string;
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ analysisData, queryResponse }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'detailed' | 'query'>('overview');

  if (!analysisData && !queryResponse) {
    return (
      <div className="p-8 text-center bg-gray-50 border border-gray-200 rounded-lg">
        <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No Analysis Data</h3>
        <p className="text-gray-600">Upload a document or submit a query to view analysis results here.</p>
      </div>
    );
  }

  const formatResponse = (response: string) => {
    return response.split('\n').filter(p => p.trim() !== '').map((paragraph, index) => (
      <p key={index} className="text-gray-700 mb-3 leading-relaxed">{paragraph}</p>
    ));
  };

  const generateChartData = () => {
    if (!analysisData) return [];
    const numerical = analysisData.analysis.numerical_intelligence;
    return [
      { name: 'Incidents', value: Math.max(0, ...(numerical.incidents || [])) },
      { name: 'Casualties', value: Math.max(0, ...(numerical.casualties || [])) },
      { name: 'Arrests', value: Math.max(0, ...(numerical.arrests || [])) },
    ].filter(item => item.value > 0);
  };

  const generateCrimeData = () => {
    if (!analysisData) return [];
    return analysisData.analysis.crime_patterns.primary_crimes.map(([crime, count]: [string, number]) => ({
      name: crime,
      value: count
    }));
  };

  const COLORS = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6'];

  const tabs = [
    { id: 'overview' as const, label: 'Overview', icon: Eye },
    { id: 'detailed' as const, label: 'Detailed Analysis', icon: Brain },
    { id: 'query' as const, label: 'Query Results', icon: FileText }
  ];

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Analysis Results</h2>
          <p className="text-gray-600 mt-1">
            {analysisData ? `Document: ${analysisData.metadata.filename}` : 'Query Response Analysis'}
          </p>
        </div>
        <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
          <Download className="h-4 w-4" />
          <span>Export Report</span>
        </button>
      </div>

      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8" aria-label="Tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab.id
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <tab.icon className="h-5 w-5" />
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
      </div>

      {activeTab === 'overview' && analysisData && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="bg-white p-6 rounded-lg shadow border">
              <p className="text-sm font-medium text-gray-600">Confidence Score</p>
              <p className="text-2xl font-bold text-blue-600">{(analysisData.analysis.confidence_score * 100).toFixed(1)}%</p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow border">
              <p className="text-sm font-medium text-gray-600">Threat Level</p>
              <p className={`text-xl font-bold ${analysisData.analysis.sentiment_analysis.threat_level === 'High' ? 'text-red-600' : 'text-orange-600'}`}>{analysisData.analysis.sentiment_analysis.threat_level}</p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow border">
              <p className="text-sm font-medium text-gray-600">Locations Found</p>
              <p className="text-2xl font-bold text-purple-600">{analysisData.analysis.geographic_intelligence.total_locations}</p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow border">
              <p className="text-sm font-medium text-gray-600">Word Count</p>
              <p className="text-2xl font-bold text-gray-600">{analysisData.analysis.text_statistics.word_count.toLocaleString()}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {generateChartData().length > 0 && (
              <div className="bg-white p-6 rounded-lg shadow border">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Metrics</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={generateChartData()} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="value" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
            {generateCrimeData().length > 0 && (
              <div className="bg-white p-6 rounded-lg shadow border">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Crime Patterns</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie data={generateCrimeData()} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} fill="#8884d8" label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                      {generateCrimeData().map((_entry, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center"><Brain className="h-5 w-5 text-blue-600 mr-2" /> AI Intelligence Summary</h3>
            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200"><p className="text-blue-800 leading-relaxed">{analysisData.analysis.intelligence_summary}</p></div>
          </div>
        </div>
      )}

      {/* <<< FIX: ADDED THIS ENTIRE BLOCK FOR THE DETAILED ANALYSIS TAB >>> */}
      {activeTab === 'detailed' && analysisData && (
        <div className="space-y-6">
          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center"><Shield className="h-5 w-5 text-blue-600 mr-2" /> Document Classification</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div><span className="font-medium text-gray-500">Primary Type:</span> <span className="text-gray-900 capitalize">{analysisData.analysis.document_classification.primary_type.replace('_', ' ')}</span></div>
              <div><span className="font-medium text-gray-500">Security Level:</span> <span className="text-red-600 font-bold">{analysisData.analysis.document_classification.security_classification}</span></div>
              <div><span className="font-medium text-gray-500">Confidence:</span> <span className="text-gray-900">{(analysisData.analysis.document_classification.confidence * 100).toFixed(1)}%</span></div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center"><Users className="h-5 w-5 text-purple-600 mr-2" /> Extracted Entities</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Object.entries(analysisData.analysis.entities).map(([type, list]) => (
                <div key={type}>
                  <h4 className="font-medium text-gray-800 capitalize mb-2 border-b pb-1">{type}</h4>
                  {list.length > 0 ? (
                    <ul className="space-y-1 text-sm text-gray-600">
                      {list.slice(0, 5).map((item) => <li key={item}>{item}</li>)}
                      {list.length > 5 && <li className="text-xs text-gray-400">...and {list.length - 5} more.</li>}
                    </ul>
                  ) : <p className="text-sm text-gray-400">None identified.</p>}
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center"><MapPin className="h-5 w-5 text-red-600 mr-2" /> Geographic Intelligence</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div><span className="font-medium text-gray-500">Identified States:</span> <span className="text-gray-900">{analysisData.analysis.geographic_intelligence.states.join(', ') || 'None'}</span></div>
                <div><span className="font-medium text-gray-500">Other Regions Mentioned:</span> <span className="text-gray-900">{analysisData.analysis.geographic_intelligence.other_locations.join(', ') || 'None'}</span></div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'query' && (
        <div className="bg-white p-6 rounded-lg shadow border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">AI Query Response</h3>
          {queryResponse ? (
             <div className="prose prose-sm max-w-none">{formatResponse(queryResponse)}</div>
          ) : (
            <p className="text-gray-500">Submit a question in the 'Query Interface' tab to see an AI-generated response here.</p>
          )}
        </div>
      )}
    </div>
  );
};

export default AnalysisResults;