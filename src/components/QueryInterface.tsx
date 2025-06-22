// src/components/QueryInterface.tsx - AI Query Interface Component (Complete Fixed Version)

import React, { useState } from 'react';
import { Search, Brain, Loader2, AlertCircle, FileText, MessageSquare, Lightbulb } from 'lucide-react';

interface QueryInterfaceProps {
  onQueryResponse?: (response: string) => void;
}

const QueryInterface: React.FC<QueryInterfaceProps> = ({ onQueryResponse }) => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [queryResponse, setQueryResponse] = useState('');
  const [error, setError] = useState<string | null>(null);

  const predefinedQueries = [
    "What are the main security threats mentioned in the documents?",
    "Summarize the geographic locations and their threat levels",
    "What criminal activities are most frequently referenced?",
    "Identify key persons and organizations mentioned",
    "What are the temporal patterns in the incidents?",
    "Provide a risk assessment based on the analyzed documents"
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query.trim() }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setQueryResponse(data.response);

      // Call parent callback if provided
      if (onQueryResponse) {
        onQueryResponse(data.response);
      }

      setError(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to process query';
      setError(errorMessage);
      setQueryResponse('');
    } finally {
      setLoading(false);
    }
  };

  const handlePredefinedQuery = (predefinedQuery: string) => {
    setQuery(predefinedQuery);
  };

  const clearQuery = () => {
    setQuery('');
    setQueryResponse('');
    setError(null);
  };

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

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2 flex items-center justify-center">
          <Brain className="h-6 w-6 text-purple-600 mr-2" />
          AI Query Interface
        </h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Ask questions about your analyzed documents using natural language.
          Get AI-powered insights and intelligence summaries.
        </p>
      </div>

      {/* Query Input Section */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <MessageSquare className="h-5 w-5 text-blue-600 mr-2" />
          Query Input
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
              placeholder="e.g., What are the main security threats in Lagos region?"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-y"
              rows={4}
              disabled={loading}
            />
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
                  Processing...
                </>
              ) : (
                <>
                  <Search className="h-5 w-5 mr-2" />
                  Submit Query
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

      {/* Predefined Queries */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Lightbulb className="h-5 w-5 text-yellow-600 mr-2" />
          Suggested Queries
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {predefinedQueries.map((predefinedQuery, index) => (
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
            <AlertCircle className="h-5 w-5 text-red-600 mr-2" />
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
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Brain className="h-5 w-5 text-green-600 mr-2" />
            AI Response
          </h3>

          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="prose prose-gray max-w-none">
              {formatResponse(queryResponse)}
            </div>
          </div>

          <div className="mt-4 flex justify-end">
            <button
              onClick={() => navigator.clipboard.writeText(queryResponse)}
              className="text-sm text-gray-600 hover:text-gray-800 flex items-center"
            >
              <FileText className="h-4 w-4 mr-1" />
              Copy Response
            </button>
          </div>
        </div>
      )}

      {/* Information Card */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start">
          <Brain className="h-5 w-5 text-blue-600 mr-2 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-blue-800">How it works</h4>
            <p className="text-sm text-blue-700 mt-1">
              This AI query interface analyzes your processed documents to provide intelligent responses.
              You can ask about threats, locations, entities, patterns, and more. The more specific your
              query, the better the AI can help you extract relevant intelligence insights.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default QueryInterface;