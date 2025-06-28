// src/components/QueryInterface.tsx - FINAL VERSION WITH TABLE SUPPORT

import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm'; // <-- 1. IMPORT THE PLUGIN
import { Search, Brain, Loader2, AlertCircle, FileText, MessageSquare, Lightbulb, Database, TrendingUp, Users, MapPin, Clock, CheckCircle, XCircle } from 'lucide-react';

interface QuerySource {
  filename: string;
  similarity: number;
  rank: number;
}

export interface QueryResponse {
  response: string;
  sources: QuerySource[];
  query: string;
  context_chunks: number;
  timestamp: string;
  model: string;
  error?: boolean;
  no_results?: boolean;
}

interface RAGStats {
  total_chunks: number;
  total_documents: number;
  index_dimension: number;
  model_name: string;
}

interface QueryInterfaceProps {
  onQueryResponse?: (response: QueryResponse) => void;
}

const QueryInterface: React.FC<QueryInterfaceProps> = ({ onQueryResponse }) => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [queryResponse, setQueryResponse] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ragStats, setRAGStats] = useState<RAGStats | null>(null);
  const [maxResults, setMaxResults] = useState(5);

  const intelligenceQueries = [
    "What are the main security threats mentioned in the documents?",
    "Summarize the geographic locations and their threat levels",
    "What criminal activities are most frequently referenced? make the response in a table form",
    "Identify key persons and organizations mentioned",
    "What are the temporal patterns in the incidents?",
    "Provide a risk assessment based on the analyzed documents",
    "What weapons or equipment are mentioned in the intelligence reports? make it a table",
    "Summarize incidents by Nigerian states and regions",
    "What are the coordination patterns between criminal groups?",
    "Analyze the threat evolution over time periods mentioned"
  ];

  useEffect(() => {
    fetchRAGStats();
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setQueryResponse(null);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          max_results: maxResults
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: QueryResponse = await response.json();
      setQueryResponse(data);

      if (onQueryResponse) {
        onQueryResponse(data);
      }

      setError(null);
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
  };

  const clearQuery = () => {
    setQuery('');
    setQueryResponse(null);
    setError(null);
  };

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 0.8) return 'text-green-600 bg-green-50';
    if (similarity >= 0.6) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2 flex items-center justify-center">
          <Brain className="h-6 w-6 text-purple-600 mr-2" />
          RAG-Powered Intelligence Query
        </h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Ask questions about your analyzed documents using advanced AI retrieval and generation.
          Get contextual answers backed by specific document sources.
        </p>
      </div>

      {/* RAG System Status */}
      {ragStats && (
        <div className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Database className="h-5 w-5 text-purple-600 mr-2" />
              <h3 className="font-semibold text-purple-800">Knowledge Base Status</h3>
            </div>
            <div className="flex items-center text-green-600">
              <CheckCircle className="h-4 w-4 mr-1" />
              <span className="text-sm font-medium">RAG System Online</span>
            </div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3 text-sm">
            <div>
              <span className="text-purple-700 font-medium">Documents:</span>
              <span className="ml-2 text-purple-900">{ragStats.total_documents.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-purple-700 font-medium">Text Chunks:</span>
              <span className="ml-2 text-purple-900">{ragStats.total_chunks.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-purple-700 font-medium">Embedding Model:</span>
              <span className="ml-2 text-purple-900">{ragStats.model_name}</span>
            </div>
            <div>
              <span className="text-purple-700 font-medium">Vector Dimension:</span>
              <span className="ml-2 text-purple-900">{ragStats.index_dimension}</span>
            </div>
          </div>
        </div>
      )}

      {/* Query Input Section */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <MessageSquare className="h-5 w-5 text-blue-600 mr-2" />
          Intelligence Query
        </h3>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
              Enter your intelligence query:
            </label>
            <textarea
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., What are the main security threats in Lagos region? Who are the key actors mentioned?"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-y"
              rows={3}
              disabled={loading}
            />
          </div>

          <div className="flex items-center space-x-4">
            <div>
              <label htmlFor="maxResults" className="block text-sm font-medium text-gray-700 mb-1">
                Max Sources:
              </label>
              <select
                id="maxResults"
                value={maxResults}
                onChange={(e) => setMaxResults(parseInt(e.target.value))}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                disabled={loading}
              >
                <option value={3}>3 Sources</option>
                <option value={5}>5 Sources</option>
                <option value={7}>7 Sources</option>
                <option value={10}>10 Sources</option>
              </select>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-3">
            <button
              type="submit"
              disabled={!query.trim() || loading}
              className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-medium py-3 px-6 rounded-lg transition-colors duration-200 flex items-center justify-center"
            >
              {loading ? (
                <>
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  Processing with RAG...
                </>
              ) : (
                <>
                  <Search className="h-5 w-5 mr-2" />
                  Query Intelligence
                </>
              )}
            </button>

            <button
              type="button"
              onClick={clearQuery}
              disabled={loading}
              className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors duration-200"
            >
              Clear
            </button>
          </div>
        </form>
      </div>

      {/* Suggested Intelligence Queries */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Lightbulb className="h-5 w-5 text-yellow-600 mr-2" />
          Intelligence Query Templates
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {intelligenceQueries.map((predefinedQuery, index) => (
            <button
              key={index}
              onClick={() => handlePredefinedQuery(predefinedQuery)}
              disabled={loading}
              className="text-left p-3 border border-gray-200 rounded-lg hover:bg-blue-50 hover:border-blue-300 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="flex items-start">
                <Search className="h-4 w-4 text-blue-600 mr-2 mt-0.5 flex-shrink-0" />
                <span className="text-sm text-gray-700">{predefinedQuery}</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <XCircle className="h-5 w-5 text-red-600 mr-2" />
            <h3 className="text-lg font-medium text-red-800">Query Error</h3>
          </div>
          <p className="text-red-700 mt-1">{error}</p>
          <button
            onClick={() => setError(null)}
            className="mt-3 text-sm text-red-600 hover:text-red-800 underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Query Response */}
      {queryResponse && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 flex items-center">
                <Brain className="h-5 w-5 text-green-600 mr-2" />
                Intelligence Analysis
              </h3>
              <div className="flex items-center text-sm text-gray-500">
                <Clock className="h-4 w-4 mr-1" />
                {new Date(queryResponse.timestamp).toLocaleString()}
              </div>
            </div>

            {queryResponse.no_results ? (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <div className="flex items-center">
                  <AlertCircle className="h-5 w-5 text-yellow-600 mr-2" />
                  <p className="text-yellow-800">{queryResponse.response}</p>
                </div>
              </div>
            ) : (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="prose prose-gray max-w-none">
                  {/* 2. USE THE PLUGIN IN THE COMPONENT */}
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {queryResponse.response}
                  </ReactMarkdown>
                </div>
              </div>
            )}

            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="flex flex-wrap items-center gap-4 text-sm text-gray-600">
                <div className="flex items-center">
                  <Database className="h-4 w-4 mr-1" />
                  {queryResponse.context_chunks} sources used
                </div>
                <div className="flex items-center">
                  <Brain className="h-4 w-4 mr-1" />
                  {queryResponse.model}
                </div>
                <button
                  onClick={() => navigator.clipboard.writeText(queryResponse.response)}
                  className="flex items-center text-blue-600 hover:text-blue-800"
                >
                  <FileText className="h-4 w-4 mr-1" />
                  Copy Response
                </button>
              </div>
            </div>
          </div>

          {queryResponse.sources && queryResponse.sources.length > 0 && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <FileText className="h-5 w-5 text-blue-600 mr-2" />
                Source Documents ({queryResponse.sources.length})
              </h3>
              <div className="space-y-3">
                {queryResponse.sources.map((source, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="flex-shrink-0">
                        <span className="inline-flex items-center justify-center h-6 w-6 rounded-full bg-blue-100 text-blue-600 text-xs font-medium">
                          {source.rank}
                        </span>
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">{source.filename}</p>
                        <p className="text-sm text-gray-500">Document source</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getSimilarityColor(source.similarity)}`}>
                        {(source.similarity * 100).toFixed(1)}% match
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* RAG Information Card */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start">
          <Brain className="h-5 w-5 text-blue-600 mr-2 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-blue-800">RAG-Powered Intelligence Analysis</h4>
            <p className="text-sm text-blue-700 mt-1">
              This system uses Retrieval-Augmented Generation (RAG) to find the most relevant information
              from your intelligence documents and generate comprehensive answers. Each response is backed
              by specific document sources with similarity scores showing relevance confidence.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default QueryInterface;