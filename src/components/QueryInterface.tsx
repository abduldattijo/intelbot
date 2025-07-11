// src/components/QueryInterface.tsx - Enhanced Multi-Crime Query Support - FIXED CRIME TYPE FILTERING

import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Search, Brain, Loader2, AlertCircle, FileText, MessageSquare, Lightbulb, Database, Zap, Target, CheckCircle, XCircle, Clock, Send, Plus, Filter } from 'lucide-react';

// Enhanced interfaces
interface RetrievedChunk {
  filename: string;
  text: string;
  relevance_score: number;
}

export interface QueryResponse {
  response: string;
  sources: Array<{ filename: string; relevance_score?: number; }>;
  query: string;
  context_chunks: number;
  timestamp: string;
  model: string;
  error?: boolean;
  no_results?: boolean;
  retrieved_chunks?: RetrievedChunk[];
  query_type?: string;
}

interface RAGStats {
  total_chunks: number;
  total_documents: number;
  bm25_docs?: number;
  index_dimension: number;
  model_name: string;
  enhancement_features?: string[];
  system_type?: string;
}

interface SystemHealth {
  status: string;
  services: {
    ollama: string;
    database: string;
    rag_system: string;
  };
  ai_model: {
    current: string;
    status: string;
  };
  data: {
    documents: number;
    chunks: number;
    bm25_docs: number;
  };
  version: string;
}

interface QueryInterfaceProps {
  onQueryResponse?: (response: QueryResponse) => void;
}

const QueryInterface: React.FC<QueryInterfaceProps> = ({ onQueryResponse }) => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [queryResponse, setQueryResponse] = useState<QueryResponse | null>(null);
  const [followUpQuestions, setFollowUpQuestions] = useState<string[]>([]);
  const [customFollowUp, setCustomFollowUp] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [ragStats, setRAGStats] = useState<RAGStats | null>(null);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [maxResults, setMaxResults] = useState(5);
  const [showCustomFollowUp, setShowCustomFollowUp] = useState(false);

  // Crime type filtering
  const [availableCrimeTypes, setAvailableCrimeTypes] = useState<string[]>([]);
  const [selectedCrimeTypeFilter, setSelectedCrimeTypeFilter] = useState<string>('');
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);

  const enhancedIntelligenceQueries = [
    "What are the main security threats mentioned in the documents?",
    "Summarize the geographic locations and their threat levels in table format",
    "What criminal activities are most frequently referenced? Create a detailed analysis",
    "Identify key persons and organizations mentioned across all reports",
    "What are the temporal patterns and trends in the incidents?",
    "Provide a comprehensive risk assessment based on the analyzed documents",
    "What weapons or equipment are mentioned? Present findings in a structured table",
    "Analyze incidents by Nigerian states and regions with threat classification",
    "What are the coordination patterns between different criminal groups?",
    "Compare incident trends between different time periods mentioned",
    "Assess the effectiveness of security responses based on the data",
    "Identify emerging threats and new criminal methodologies",
    // Crime-specific queries
    "Compare Armed Banditry incidents vs Kidnapping incidents by month",
    "What are the patterns in Cult Activities across different states?",
    "Analyze the seasonal trends in Farmers/Herders Clashes",
    "How do Boko Haram activities correlate with security responses?",
    "What crime types show the most concerning growth trends?"
  ];

  useEffect(() => {
    fetchRAGStats();
    fetchSystemHealth();
    fetchCrimeTypes();
  }, []);

  const fetchRAGStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/rag-stats');
      if (response.ok) {
        const stats = await response.json();
        setRAGStats(stats);
      }
    } catch (err) {
      console.error('Failed to fetch RAG stats:', err);
    }
  };

  const fetchSystemHealth = async () => {
    try {
      const response = await fetch('http://localhost:8000/system-health');
      if (response.ok) {
        const health = await response.json();
        setSystemHealth(health);
      }
    } catch (err) {
      console.error('Failed to fetch system health:', err);
    }
  };

  // Fetch available crime types for filtering
  const fetchCrimeTypes = async () => {
    try {
      const response = await fetch('http://localhost:8000/crime-types');
      if (response.ok) {
        const data = await response.json();
        setAvailableCrimeTypes(data.crime_types || []);
      }
    } catch (err) {
      console.error('Failed to fetch crime types:', err);
    }
  };

  const fetchFollowUps = async (originalQuery: string, originalResponse: string) => {
    try {
        const response = await fetch('http://localhost:8000/generate-followups', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: originalQuery, response: originalResponse }),
        });
        if (response.ok) {
            const questions = await response.json();
            setFollowUpQuestions(questions);
        }
    } catch (err) {
        console.error('Failed to fetch follow-up questions:', err);
        setFollowUpQuestions([
          "What additional threat indicators should be monitored?",
          "How do these findings compare to historical patterns?",
          "What are the operational implications for security planning?"
        ]);
    }
  };

  const handleSubmit = async (e: React.FormEvent, queryText?: string) => {
    e.preventDefault();
    const actualQuery = queryText || query.trim();
    if (!actualQuery) return;

    setLoading(true);
    setError(null);
    setQueryResponse(null);
    setFollowUpQuestions([]);
    setShowCustomFollowUp(false);

    try {
      // FIXED: Send crime type as separate parameter instead of modifying query text
      const requestBody = {
        query: actualQuery, // Send original query without modification
        max_results: maxResults,
        crime_type: selectedCrimeTypeFilter || null // Send crime type as separate parameter
      };

      console.log('Sending request with crime type filter:', requestBody);

      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
      }

      const data: QueryResponse = await response.json();
      setQueryResponse(data);

      if (onQueryResponse) {
        onQueryResponse(data);
      }

      if (!data.error && !data.no_results) {
        await fetchFollowUps(actualQuery, data.response);
      }

      setError(null);

      if (queryText) {
        setCustomFollowUp('');
      }

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to process query';
      setError(errorMessage);
      setQueryResponse(null);
    } finally {
      setLoading(false);
    }
  };

  const handlePredefinedQuery = (predefinedQuery: string) => {
    setQuery(predefinedQuery);
    setQueryResponse(null);
    setFollowUpQuestions([]);
    setError(null);
    setShowCustomFollowUp(false);
  };

  const handleFollowUpClick = (question: string) => {
    setCustomFollowUp(question);
    setShowCustomFollowUp(true);
  };

  const handleCustomFollowUpSubmit = (e: React.FormEvent) => {
    if (customFollowUp.trim()) {
      handleSubmit(e, customFollowUp.trim());
    }
  };

  const clearQuery = () => {
    setQuery('');
    setQueryResponse(null);
    setFollowUpQuestions([]);
    setError(null);
    setCustomFollowUp('');
    setShowCustomFollowUp(false);
    setSelectedCrimeTypeFilter('');

    setTimeout(() => {
      const queryInput = document.getElementById('query');
      if (queryInput) {
        queryInput.focus();
      }
    }, 100);
  };

  const getRelevanceColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-100 border-green-200';
    if (score >= 0.5) return 'text-yellow-600 bg-yellow-100 border-yellow-200';
    return 'text-red-600 bg-red-100 border-red-200';
  };

  const getSystemHealthColor = (status: string) => {
    if (status === 'healthy') return 'text-green-600';
    if (status === 'degraded') return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="p-6 space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2 flex items-center justify-center">
          <Brain className="h-6 w-6 text-purple-600 mr-2" />
          Multi-Crime Intelligence Query System
        </h2>
        <p className="text-gray-600 max-w-3xl mx-auto">
          AI-powered document analysis with Gemma2:9B, hybrid search, crime type filtering, and custom follow-up capabilities.
        </p>
      </div>

      {/* System Health Status */}
      {systemHealth && (
        <div className="bg-gradient-to-r from-purple-50 to-blue-50 border-2 border-purple-200 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <Zap className="h-6 w-6 text-purple-600 mr-2" />
              <h3 className="font-bold text-purple-800 text-lg">Multi-Crime System Status</h3>
            </div>
            <div className={`flex items-center ${getSystemHealthColor(systemHealth.status)}`}>
              <CheckCircle className="h-5 w-5 mr-1" />
              <span className="font-medium capitalize">{systemHealth.status}</span>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-white rounded-lg p-3 border border-purple-200">
              <div className="text-sm text-purple-700 font-medium">AI Model</div>
              <div className="text-lg font-bold text-purple-900">{systemHealth.ai_model.current}</div>
            </div>
            <div className="bg-white rounded-lg p-3 border border-purple-200">
              <div className="text-sm text-purple-700 font-medium">Documents</div>
              <div className="text-xl font-bold text-purple-900">{systemHealth.data.documents}</div>
            </div>
            <div className="bg-white rounded-lg p-3 border border-purple-200">
              <div className="text-sm text-purple-700 font-medium">Chunks</div>
              <div className="text-xl font-bold text-purple-900">{systemHealth.data.chunks}</div>
            </div>
            <div className="bg-white rounded-lg p-3 border border-purple-200">
              <div className="text-sm text-purple-700 font-medium">Crime Types</div>
              <div className="text-xl font-bold text-purple-900">{availableCrimeTypes.length}</div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="flex items-center">
              <span className="text-purple-700 font-medium">Ollama Service:</span>
              <span className={`ml-2 px-2 py-1 rounded-full text-xs font-bold ${
                systemHealth.services.ollama === 'running' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                {systemHealth.services.ollama}
              </span>
            </div>
            <div className="flex items-center">
              <span className="text-purple-700 font-medium">AI Model:</span>
              <span className={`ml-2 px-2 py-1 rounded-full text-xs font-bold ${
                systemHealth.ai_model.status === 'available' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                {systemHealth.ai_model.status}
              </span>
            </div>
            <div className="flex items-center">
              <span className="text-purple-700 font-medium">Database:</span>
              <span className="ml-2 px-2 py-1 rounded-full text-xs font-bold bg-green-100 text-green-800">
                {systemHealth.services.database}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Main Query Interface */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <MessageSquare className="h-5 w-5 text-blue-600 mr-2" />
          Multi-Crime Intelligence Query
        </h3>

        {/* Advanced Filters */}
        <div className="mb-4">
          <button
            onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
            className="flex items-center text-sm text-blue-600 hover:text-blue-800 mb-3"
          >
            <Filter className="h-4 w-4 mr-1" />
            {showAdvancedFilters ? 'Hide' : 'Show'} Advanced Filters
          </button>

          {showAdvancedFilters && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-4">
              <h4 className="text-sm font-medium text-gray-700 mb-3">Query Filters</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="crimeTypeFilter" className="block text-sm font-medium text-gray-700 mb-1">
                    Focus on Crime Type (Optional)
                  </label>
                  <select
                    id="crimeTypeFilter"
                    value={selectedCrimeTypeFilter}
                    onChange={(e) => setSelectedCrimeTypeFilter(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                  >
                    <option value="">All Crime Types</option>
                    {availableCrimeTypes.map(type => (
                      <option key={type} value={type}>{type}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label htmlFor="maxResults" className="block text-sm font-medium text-gray-700 mb-1">
                    Max Sources
                  </label>
                  <select
                    id="maxResults"
                    value={maxResults}
                    onChange={(e) => setMaxResults(parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                  >
                    <option value={3}>3 (Fast)</option>
                    <option value={5}>5 (Balanced)</option>
                    <option value={7}>7 (Comprehensive)</option>
                    <option value={10}>10 (Deep Analysis)</option>
                    <option value={15}>15 (Extensive)</option>
                    <option value={20}>20 (Maximum Coverage)</option>
                  </select>
                </div>
              </div>
              {selectedCrimeTypeFilter && (
                <div className="mt-3 p-2 bg-blue-50 border border-blue-200 rounded text-sm text-blue-800">
                  <strong>Filter Active:</strong> Queries will focus on {selectedCrimeTypeFilter} documents and context.
                </div>
              )}
            </div>
          )}
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <textarea
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., Analyze the geographic distribution of kidnapping incidents and compare with armed banditry patterns..."
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-y"
            rows={3}
            disabled={loading}
          />

          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">
              {availableCrimeTypes.length > 0 && (
                <span>
                  {availableCrimeTypes.length} crime types available for analysis
                  {selectedCrimeTypeFilter && ` • Filtering by: ${selectedCrimeTypeFilter}`}
                </span>
              )}
            </div>
            <div className="flex gap-3">
              <button
                type="button"
                onClick={clearQuery}
                disabled={loading}
                className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                title="Clear query and start fresh"
              >
                Clear All
              </button>
              <button
                type="submit"
                disabled={!query.trim() || loading}
                className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 disabled:from-gray-400 disabled:to-gray-400 text-white font-medium py-3 px-6 rounded-lg flex items-center justify-center"
              >
                {loading ? <Loader2 className="h-5 w-5 mr-2 animate-spin" /> : <Zap className="h-5 w-5 mr-2" />}
                {loading ? 'Processing...' : 'Run Analysis'}
              </button>
            </div>
          </div>
        </form>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <XCircle className="h-5 w-5 text-red-600 mr-2" />
            <h3 className="text-lg font-medium text-red-800">Query Error:</h3>
            <p className="text-red-700 ml-2">{error}</p>
          </div>
        </div>
      )}

      {queryResponse && (
        <div className="space-y-6">
          {/* Main Response */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 flex items-center">
                <Brain className="h-5 w-5 text-green-600 mr-2" />
                Multi-Crime Intelligence Analysis
                {queryResponse.query_type && (
                  <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                    {queryResponse.query_type}
                  </span>
                )}
                {selectedCrimeTypeFilter && (
                  <span className="ml-2 px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded-full">
                    {selectedCrimeTypeFilter}
                  </span>
                )}
              </h3>
              <div className="flex items-center space-x-4 text-sm text-gray-500">
                <div className="flex items-center">
                  <Clock className="h-4 w-4 mr-1" />
                  {new Date(queryResponse.timestamp).toLocaleString()}
                </div>
                <div className="flex items-center">
                  <FileText className="h-4 w-4 mr-1" />
                  {queryResponse.context_chunks} sources
                </div>
              </div>
            </div>

            {queryResponse.no_results ? (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 flex items-center">
                <AlertCircle className="h-5 w-5 text-yellow-600 mr-2" />
                <p className="text-yellow-800">{queryResponse.response}</p>
              </div>
            ) : (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4 prose prose-sm max-w-none prose-p:text-gray-800">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{queryResponse.response}</ReactMarkdown>
              </div>
            )}
          </div>

          {/* Enhanced Follow-up Questions Section */}
          {(followUpQuestions.length > 0 || queryResponse.context_chunks > 0) && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Lightbulb className="h-5 w-5 text-yellow-500 mr-2" />
                Follow-up Questions
              </h3>

              {/* AI-Generated Follow-ups */}
              {followUpQuestions.length > 0 && (
                <div className="mb-4">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">AI-Generated Suggestions:</h4>
                  <div className="flex flex-wrap gap-2">
                    {followUpQuestions.map((q, index) => (
                      <button
                        key={index}
                        onClick={() => handleFollowUpClick(q)}
                        className="p-2 px-3 text-sm border border-gray-200 rounded-full hover:bg-blue-50 hover:border-blue-300 transition-colors text-left"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Custom Follow-up Input */}
              <div className="border-t pt-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-medium text-gray-700">Ask Your Own Follow-up:</h4>
                  <button
                    onClick={() => setShowCustomFollowUp(!showCustomFollowUp)}
                    className="flex items-center text-sm text-blue-600 hover:text-blue-800"
                  >
                    <Plus className="h-4 w-4 mr-1" />
                    Custom Question
                  </button>
                </div>

                {(showCustomFollowUp || customFollowUp) && (
                  <form onSubmit={handleCustomFollowUpSubmit} className="flex gap-2">
                    <input
                      type="text"
                      value={customFollowUp}
                      onChange={(e) => setCustomFollowUp(e.target.value)}
                      placeholder="Ask a custom follow-up question..."
                      className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      disabled={loading}
                    />
                    <button
                      type="submit"
                      disabled={!customFollowUp.trim() || loading}
                      className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 flex items-center"
                    >
                      {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                    </button>
                  </form>
                )}
              </div>
            </div>
          )}

          {/* Source Evidence */}
          {queryResponse.retrieved_chunks && queryResponse.retrieved_chunks.length > 0 && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <FileText className="h-5 w-5 text-blue-600 mr-2" />
                Intelligence Sources ({queryResponse.retrieved_chunks.length})
                {selectedCrimeTypeFilter && (
                  <span className="ml-2 text-sm text-purple-600">• Filtered by {selectedCrimeTypeFilter}</span>
                )}
              </h3>
              <div className="space-y-4">
                {queryResponse.retrieved_chunks.map((chunk, index) => (
                  <details key={index} className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                    <summary className="font-medium text-gray-800 cursor-pointer flex justify-between items-center">
                      <div>
                        Source: <span className="text-blue-700">{chunk.filename}</span>
                      </div>
                      <div className={`px-2 py-1 text-xs font-bold rounded-md border ${getRelevanceColor(chunk.relevance_score)}`}>
                        {(chunk.relevance_score * 100).toFixed(1)}% Relevance
                      </div>
                    </summary>
                    <div className="mt-4 pt-4 border-t border-gray-200 text-sm text-gray-700 bg-white p-3 rounded">
                      "{chunk.text}"
                    </div>
                  </details>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Query Templates */}
      {!queryResponse && !loading && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Target className="h-5 w-5 text-blue-600 mr-2" />
              Multi-Crime Intelligence Query Templates
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {enhancedIntelligenceQueries.map((predefinedQuery, index) => (
                <button
                  key={index}
                  onClick={() => handlePredefinedQuery(predefinedQuery)}
                  disabled={loading}
                  className="text-left p-4 border border-gray-200 rounded-lg hover:bg-blue-50 hover:border-blue-300 transition-colors duration-200"
                >
                  <div className="flex items-start">
                    <Search className="h-4 w-4 text-blue-600 mr-3 mt-0.5 flex-shrink-0" />
                    <span className="text-sm text-gray-700">{predefinedQuery}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
      )}

    </div>
  );
};

export default QueryInterface;