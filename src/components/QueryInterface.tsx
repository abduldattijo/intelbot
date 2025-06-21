// src/components/QueryInterface.tsx - Enhanced Query Interface Component

import React, { useState } from 'react';
import { Search, Send, Brain, Lightbulb, MessageSquare, Loader } from 'lucide-react';

interface QueryInterfaceProps {
  onQueryResponse: (response: string) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

const QueryInterface: React.FC<QueryInterfaceProps> = ({
  onQueryResponse,
  isLoading,
  setIsLoading
}) => {
  const [query, setQuery] = useState('');
  const [queryHistory, setQueryHistory] = useState<Array<{query: string, response: string, timestamp: Date}>>([]);

  const suggestedQueries = [
    "Show me a summary of all incidents",
    "What are the geographic patterns?",
    "Analyze crime trends over time",
    "What's the casualty rate by location?",
    "Show me forecasted incidents for next quarter",
    "Which areas have the highest threat levels?",
    "What are the most common crime types?",
    "How effective are arrest rates?"
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!query.trim()) {
      alert('Please enter a query');
      return;
    }

    try {
      setIsLoading(true);

      const response = await fetch('http://localhost:8000/query-documents', {
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

      // Add to history
      setQueryHistory(prev => [{
        query: query.trim(),
        response: data.response,
        timestamp: new Date()
      }, ...prev]);

      // Notify parent
      onQueryResponse(data.response);

      // Clear query
      setQuery('');

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to process query';
      onQueryResponse(`Error: ${errorMessage}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestedQuery = (suggestedQuery: string) => {
    setQuery(suggestedQuery);
  };

  const formatResponse = (response: string) => {
    // Split response into sections and format
    const sections = response.split('\n\n');
    return sections.map((section, index) => {
      if (section.startsWith('**') && section.endsWith('**')) {
        // Header
        return (
          <h4 key={index} className="font-semibold text-gray-900 mt-4 mb-2">
            {section.replace(/\*\*/g, '')}
          </h4>
        );
      } else if (section.includes('•')) {
        // Bullet points
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
        // Regular paragraph
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
          <Brain className="h-6 w-6 text-green-600 mr-2" />
          Intelligent Query Interface
        </h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Ask questions about your processed documents. Our AI will analyze patterns,
          extract insights, and provide comprehensive intelligence reports.
        </p>
      </div>

      {/* Query Form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about incidents, patterns, trends, forecasts, or specific locations..."
            className="w-full pl-10 pr-12 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent text-gray-900"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !query.trim()}
            className="absolute right-2 top-1/2 transform -translate-y-1/2 p-1.5 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <Loader className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </button>
        </div>
      </form>

      {/* Suggested Queries */}
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <div className="flex items-center mb-3">
          <Lightbulb className="h-4 w-4 text-green-600 mr-2" />
          <h3 className="text-sm font-semibold text-green-800">Suggested Queries:</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {suggestedQueries.map((suggestedQuery, index) => (
            <button
              key={index}
              onClick={() => handleSuggestedQuery(suggestedQuery)}
              disabled={isLoading}
              className="text-left p-2 text-sm text-green-700 hover:bg-green-100 rounded border border-green-200 transition-colors disabled:opacity-50"
            >
              {suggestedQuery}
            </button>
          ))}
        </div>
      </div>

      {/* Query History */}
      {queryHistory.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <MessageSquare className="h-5 w-5 text-blue-600 mr-2" />
            Query History
          </h3>

          <div className="space-y-4 max-h-96 overflow-y-auto">
            {queryHistory.map((item, index) => (
              <div key={index} className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                <div className="mb-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-blue-600">Query:</span>
                    <span className="text-xs text-gray-500">
                      {item.timestamp.toLocaleString()}
                    </span>
                  </div>
                  <p className="text-gray-900 bg-gray-50 p-2 rounded border">
                    {item.query}
                  </p>
                </div>

                <div>
                  <span className="text-sm font-medium text-green-600 mb-2 block">Response:</span>
                  <div className="text-gray-700 prose prose-sm max-w-none">
                    {formatResponse(item.response)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tips */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-800 mb-2">💡 Query Tips:</h3>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• Be specific about what information you're looking for</li>
          <li>• Ask about trends, patterns, or comparisons between different areas</li>
          <li>• Use terms like "analyze", "compare", "forecast" for detailed insights</li>
          <li>• You can ask about specific time periods, locations, or crime types</li>
        </ul>
      </div>
    </div>
  );
};

export default QueryInterface;