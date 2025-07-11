// src/components/ComparisonDashboard.tsx - Enhanced with Multi-Crime Type Support

import React, { useState, useEffect } from 'react';
import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { BarChart3, ArrowRight, Brain, AlertTriangle, Loader2, Table, TrendingUp, Database, RefreshCw, Target, Filter } from 'lucide-react';

// Interfaces
interface MonthlyComparisonData {
  metric: string;
  value1: string;
  value2: string;
  change: string;
}

interface ComparisonResponse {
  month1: string;
  month2: string;
  comparison_table: MonthlyComparisonData[];
  ai_inference: string;
}

interface DebugInfo {
  total_rows?: number;
  raw_dates_sample?: string[];
  formatted_months_count?: number;
  error?: string;
}

const ComparisonDashboard: React.FC = () => {
  const [availableMonths, setAvailableMonths] = useState<string[]>([]);
  const [availableCrimeTypes, setAvailableCrimeTypes] = useState<string[]>([]);
  const [selectedCrimeType, setSelectedCrimeType] = useState<string>('');
  const [month1, setMonth1] = useState<string>('');
  const [month2, setMonth2] = useState<string>('');
  const [comparisonData, setComparisonData] = useState<ComparisonResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'table' | 'chart'>('table');
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);
  const [fetchingMonths, setFetchingMonths] = useState(false);
  const [fetchingCrimeTypes, setFetchingCrimeTypes] = useState(true);

  useEffect(() => {
    fetchCrimeTypes();
  }, []);

  useEffect(() => {
    if (selectedCrimeType) {
      fetchAvailableMonths();
    }
  }, [selectedCrimeType]);

  const fetchCrimeTypes = async () => {
    try {
      setFetchingCrimeTypes(true);
      setError(null);

      console.log('Fetching available crime types...');
      const response = await fetch('http://localhost:8000/crime-types');

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Crime types API response:', data);

      const crimeTypes = data.crime_types || [];
      setAvailableCrimeTypes(crimeTypes);

      if (crimeTypes.length > 0) {
        setSelectedCrimeType(crimeTypes[0]); // Auto-select first crime type
        console.log('Auto-selected crime type:', crimeTypes[0]);
      } else {
        setError("No crime types available. Please upload some documents first.");
      }

    } catch (err) {
      console.error('Failed to fetch crime types:', err);
      setError(`Could not load crime types: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setAvailableCrimeTypes([]);
    } finally {
      setFetchingCrimeTypes(false);
    }
  };

  const fetchAvailableMonths = async () => {
    if (!selectedCrimeType) return;

    try {
      setFetchingMonths(true);
      setError(null);

      console.log('Fetching available months for crime type:', selectedCrimeType);

      const url = `http://localhost:8000/available-months?crime_type=${encodeURIComponent(selectedCrimeType)}`;
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Months API response:', data);

      const months = data.available_months || [];
      setAvailableMonths(months);
      setDebugInfo(data.debug_info);

      if (months.length >= 2) {
        setMonth1(months[0]); // Most recent
        setMonth2(months[1]); // Second most recent
        console.log('Auto-selected months:', months[0], 'and', months[1]);
      } else if (months.length === 0) {
        setError(`No months available for ${selectedCrimeType}. Try a different crime type.`);
      } else {
        setError(`Only one month available for ${selectedCrimeType}. Need at least 2 months to compare.`);
      }

    } catch (err) {
      console.error('Failed to fetch months:', err);
      setError(`Could not load available months: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setAvailableMonths([]);
    } finally {
      setFetchingMonths(false);
    }
  };

  const handleCompare = async () => {
    if (!month1 || !month2 || !selectedCrimeType) {
      setError('Please select crime type and two different months to compare.');
      return;
    }
    if (month1 === month2) {
      setError('Please select two different months to compare.');
      return;
    }

    setLoading(true);
    setError(null);
    setComparisonData(null);

    try {
      console.log(`Comparing ${month1} vs ${month2} for ${selectedCrimeType}`);

      const url = `http://localhost:8000/compare-months?month1=${month1}&month2=${month2}&crime_type=${encodeURIComponent(selectedCrimeType)}`;
      const response = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'}
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
        throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
      }

      const data: ComparisonResponse = await response.json();
      console.log('Comparison result:', data);
      setComparisonData(data);

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred.';
      console.error('Comparison failed:', errorMessage);
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const getChangeColor = (change: string) => {
    if (change.startsWith('+')) return 'text-red-600 bg-red-50';
    if (change.startsWith('-')) return 'text-green-600 bg-green-50';
    return 'text-gray-500 bg-gray-50';
  };

  const formatAIResponse = (text: string) => {
    if (!text) return <div className="text-gray-500">No analysis available.</div>;

    let cleanText = text
      .replace(/\*\*([^*]+)\*\*/g, '$1')
      .replace(/\*([^*]+)\*/g, '$1')
      .replace(/#{1,6}\s*/g, '')
      .replace(/`([^`]+)`/g, '$1')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/^\s*[-*+]\s+/gm, '')
      .replace(/^\s*\d+\.\s+/gm, '')
      .trim();

    const sections = cleanText.split(/(?=EXECUTIVE SUMMARY|TREND ANALYSIS|CAUSAL FACTORS|STRATEGIC IMPLICATIONS|RECOMMENDATIONS)/i);

    return (
      <div className="space-y-4">
        {sections.map((section, index) => {
          if (!section.trim()) return null;

          const lines = section.trim().split('\n').filter(line => line.trim());
          if (lines.length === 0) return null;

          const firstLine = lines[0].trim();
          const isHeader = /^(EXECUTIVE SUMMARY|TREND ANALYSIS|CAUSAL FACTORS|STRATEGIC IMPLICATIONS|RECOMMENDATIONS)/i.test(firstLine);

          if (isHeader) {
            const headerText = firstLine.replace(/[:\-_=]/g, '').trim();
            const contentLines = lines.slice(1);

            return (
              <div key={index} className="mb-6">
                <div className="flex items-center mb-3">
                  <div className="h-px bg-indigo-200 flex-grow"></div>
                  <h4 className="px-3 font-bold text-indigo-900 text-sm uppercase tracking-wide bg-indigo-50 rounded-full py-1">
                    {headerText}
                  </h4>
                  <div className="h-px bg-indigo-200 flex-grow"></div>
                </div>
                <div className="text-indigo-800 text-sm leading-relaxed space-y-2">
                  {contentLines.map((line, lineIndex) => {
                    const cleanLine = line.trim();
                    if (!cleanLine) return null;

                    if (/^\d+\.?\s*/.test(cleanLine)) {
                      const number = cleanLine.match(/^(\d+)\.?\s*/)?.[1];
                      const content = cleanLine.replace(/^\d+\.?\s*/, '');
                      return (
                        <div key={lineIndex} className="flex items-start space-x-3 p-2 bg-white rounded border-l-3 border-indigo-300">
                          <span className="flex-shrink-0 w-6 h-6 bg-indigo-600 text-white text-xs font-bold rounded-full flex items-center justify-center">
                            {number}
                          </span>
                          <p className="flex-1">{content}</p>
                        </div>
                      );
                    }

                    return (
                      <p key={lineIndex} className="text-gray-700">
                        {cleanLine}
                      </p>
                    );
                  })}
                </div>
              </div>
            );
          } else {
            return (
              <div key={index} className="text-indigo-800 text-sm leading-relaxed">
                {lines.map((line, lineIndex) => (
                  <p key={lineIndex} className="mb-2 text-gray-700">
                    {line.trim()}
                  </p>
                ))}
              </div>
            );
          }
        }).filter(Boolean)}
      </div>
    );
  };

  const prepareChartData = () => {
    if (!comparisonData) return [];

    return comparisonData.comparison_table
      .filter(row => row.metric !== 'Threat Level')
      .map(row => {
        const parseNumber = (str: string) => {
          const cleaned = str.replace(/,/g, '');
          return parseInt(cleaned) || 0;
        };

        return {
          metric: row.metric.replace('Total ', ''),
          [comparisonData.month2]: parseNumber(row.value2),
          [comparisonData.month1]: parseNumber(row.value1),
          change: row.change
        };
      });
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-medium text-gray-900">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.dataKey}: {entry.value.toLocaleString()}
            </p>
          ))}
          <p className="text-xs text-gray-600 mt-1 font-medium">
            Change: {data.change}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 flex items-center">
          <BarChart3 className="h-6 w-6 text-indigo-600 mr-2" />
          Monthly Intelligence Comparison
        </h2>
        <p className="text-gray-600 mt-1">
          Compare monthly metrics for specific crime types and generate AI-powered strategic insights.
        </p>
      </div>

      {/* Crime Type Selector - NEW */}
      <div className="bg-white p-6 rounded-lg shadow border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Target className="h-5 w-5 text-blue-600 mr-2" />
          Step 1: Select Crime Type
        </h3>
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <label htmlFor="crimeType" className="block text-sm font-medium text-gray-700 mb-2">
              Crime Category
              {fetchingCrimeTypes && <Loader2 className="inline h-4 w-4 ml-2 animate-spin" />}
            </label>
            <select
              id="crimeType"
              value={selectedCrimeType}
              onChange={(e) => setSelectedCrimeType(e.target.value)}
              className="w-full pl-3 pr-10 py-3 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
              disabled={fetchingCrimeTypes || availableCrimeTypes.length === 0}
            >
              <option value="" disabled>
                {fetchingCrimeTypes ? 'Loading crime types...' : availableCrimeTypes.length === 0 ? 'No crime types available' : 'Select a crime type...'}
              </option>
              {availableCrimeTypes.map(type => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>
          <button
            onClick={fetchCrimeTypes}
            disabled={fetchingCrimeTypes}
            className="px-4 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:bg-gray-100 mt-6"
            title="Refresh crime types"
          >
            <RefreshCw className={`h-5 w-5 ${fetchingCrimeTypes ? 'animate-spin' : ''}`} />
          </button>
        </div>
        {selectedCrimeType && (
          <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">
              <span className="font-medium">Selected:</span> {selectedCrimeType}
            </p>
          </div>
        )}
      </div>

      {/* Month Selection - Updated */}
      <div className="bg-white p-6 rounded-lg shadow border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Filter className="h-5 w-5 text-purple-600 mr-2" />
          Step 2: Select Months to Compare
        </h3>
        <div className="flex flex-col md:flex-row items-center gap-4">
          <div className="flex-1 w-full">
            <label htmlFor="month2" className="block text-sm font-medium text-gray-700">
              Baseline Month (Older)
              {fetchingMonths && <Loader2 className="inline h-4 w-4 ml-2 animate-spin" />}
            </label>
            <select
              id="month2"
              value={month2}
              onChange={(e) => setMonth2(e.target.value)}
              className="mt-1 block w-full pl-3 pr-10 py-3 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
              disabled={!selectedCrimeType || fetchingMonths || availableMonths.length === 0}
            >
              <option value="" disabled>
                {!selectedCrimeType ? 'Select crime type first' :
                 fetchingMonths ? 'Loading months...' :
                 availableMonths.length === 0 ? 'No months available' : 'Select baseline month'}
              </option>
              {availableMonths.map(m => (
                <option key={m} value={m}>
                  {new Date(m).toLocaleString('default', { month: 'long', year: 'numeric' })}
                </option>
              ))}
            </select>
          </div>

          <ArrowRight className="h-6 w-6 text-gray-400 mt-6 hidden md:block" />

          <div className="flex-1 w-full">
            <label htmlFor="month1" className="block text-sm font-medium text-gray-700">Comparison Month (Newer)</label>
            <select
              id="month1"
              value={month1}
              onChange={(e) => setMonth1(e.target.value)}
              className="mt-1 block w-full pl-3 pr-10 py-3 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
              disabled={!selectedCrimeType || fetchingMonths || availableMonths.length === 0}
            >
              <option value="" disabled>
                {!selectedCrimeType ? 'Select crime type first' :
                 fetchingMonths ? 'Loading months...' :
                 availableMonths.length === 0 ? 'No months available' : 'Select comparison month'}
              </option>
              {availableMonths.map(m => (
                <option key={m} value={m}>
                  {new Date(m).toLocaleString('default', { month: 'long', year: 'numeric' })}
                </option>
              ))}
            </select>
          </div>

          <div className="flex gap-2">
            <button
              onClick={fetchAvailableMonths}
              disabled={!selectedCrimeType || fetchingMonths}
              className="px-3 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:bg-gray-100 mt-6"
              title="Refresh months list"
            >
              <RefreshCw className={`h-5 w-5 ${fetchingMonths ? 'animate-spin' : ''}`} />
            </button>

            <button
              onClick={handleCompare}
              disabled={loading || !month1 || !month2 || !selectedCrimeType || month1 === month2 || fetchingMonths}
              className="px-8 py-3 bg-indigo-600 text-white font-semibold rounded-lg shadow-md hover:bg-indigo-700 disabled:bg-gray-400 transition-colors flex items-center justify-center mt-6"
            >
              {loading ? <Loader2 className="h-5 w-5 animate-spin mr-2" /> : <Brain className="h-5 w-5 mr-2" />}
              {loading ? 'Analyzing...' : 'Compare & Analyze'}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 text-red-600 mr-2" />
            <h3 className="text-red-800 font-medium">Error</h3>
          </div>
          <p className="text-red-700 mt-1">{error}</p>
          {(availableCrimeTypes.length === 0 || availableMonths.length === 0) && (
            <div className="mt-3 text-sm text-red-600">
              <p><strong>Troubleshooting:</strong></p>
              <ul className="list-disc list-inside mt-1 space-y-1">
                <li>Make sure you've uploaded intelligence reports with different crime types</li>
                <li>Check that the backend server is running on http://localhost:8000</li>
                <li>Verify documents contain proper "RETURNS ON [CRIME TYPE] FOR..." headers</li>
                <li>Try refreshing the crime types and months lists</li>
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Comparison Results */}
      {comparisonData && (
        <div className="space-y-6">
          {/* Summary Card */}
          <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-indigo-900 mb-2">Comparison Summary</h3>
            <p className="text-indigo-800">
              Analyzing <span className="font-bold">{selectedCrimeType}</span> trends between{' '}
              <span className="font-bold">{new Date(comparisonData.month2).toLocaleString('default', { month: 'long', year: 'numeric' })}</span> and{' '}
              <span className="font-bold">{new Date(comparisonData.month1).toLocaleString('default', { month: 'long', year: 'numeric' })}</span>
            </p>
          </div>

          <div className="bg-white rounded-lg shadow border p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Quantitative Comparison</h3>
              <div className="flex bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => setViewMode('table')}
                  className={`flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                    viewMode === 'table' 
                      ? 'bg-white text-indigo-600 shadow-sm' 
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <Table className="h-4 w-4 mr-1" />
                  Table
                </button>
                <button
                  onClick={() => setViewMode('chart')}
                  className={`flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                    viewMode === 'chart' 
                      ? 'bg-white text-indigo-600 shadow-sm' 
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <TrendingUp className="h-4 w-4 mr-1" />
                  Chart
                </button>
              </div>
            </div>

            {viewMode === 'table' && (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Metric</th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">{comparisonData.month2}</th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">{comparisonData.month1}</th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Change</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {comparisonData.comparison_table.map((row) => (
                      <tr key={row.metric}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{row.metric}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{row.value2}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{row.value1}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-bold">
                          <span className={`px-2 py-1 rounded-md ${getChangeColor(row.change)}`}>
                            {row.change}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {viewMode === 'chart' && (
              <div className="space-y-6">
                <div style={{ height: '400px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsBarChart
                      data={prepareChartData()}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                      barGap={10}
                    >
                      <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                      <XAxis
                        dataKey="metric"
                        tick={{ fontSize: 12 }}
                        interval={0}
                        angle={-45}
                        textAnchor="end"
                        height={60}
                      />
                      <YAxis
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => value.toLocaleString()}
                      />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend />
                      <Bar
                        dataKey={comparisonData.month2}
                        fill="#6366f1"
                        name={comparisonData.month2}
                        radius={[2, 2, 0, 0]}
                      />
                      <Bar
                        dataKey={comparisonData.month1}
                        fill="#ec4899"
                        name={comparisonData.month1}
                        radius={[2, 2, 0, 0]}
                      />
                    </RechartsBarChart>
                  </ResponsiveContainer>
                </div>

                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-gray-900 mb-2">Chart Insights</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-gray-600">
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-indigo-500 rounded mr-2"></div>
                      <span>{comparisonData.month2} (Baseline)</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-pink-500 rounded mr-2"></div>
                      <span>{comparisonData.month1} (Comparison)</span>
                    </div>
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    Hover over bars to see detailed values and percentage changes for <strong>{selectedCrimeType}</strong>
                  </p>
                </div>
              </div>
            )}
          </div>

          <div className="bg-white rounded-lg shadow border p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Brain className="h-5 w-5 text-indigo-600 mr-2" />
              AI-Powered Strategic Inference
            </h3>
            <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
              {formatAIResponse(comparisonData.ai_inference)}
            </div>
          </div>
        </div>
      )}

      {!comparisonData && !loading && !error && availableCrimeTypes.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-start">
            <Brain className="h-5 w-5 text-blue-600 mr-2 mt-0.5" />
            <div>
              <h4 className="text-sm font-medium text-blue-800">Ready to Compare</h4>
              <p className="text-sm text-blue-700 mt-1">
                {availableCrimeTypes.length} crime types available. Select a crime type, choose two different months, and click "Compare & Analyze" to generate insights.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ComparisonDashboard;