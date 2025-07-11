// src/components/AnalysisResults.tsx - Enhanced with Multi-Crime Database Charts

import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, Area, AreaChart } from 'recharts';
import { FileText, MapPin, Calendar, Users, Shield, AlertTriangle, Brain, Eye, Download, TrendingUp, Database, BarChart3, Filter, Target } from 'lucide-react';
import { IntelligenceDocument } from '../App';

interface AnalysisResultsProps {
  analysisData: IntelligenceDocument | null;
  queryResponse: string;
}

interface MonthlyData {
  month: string;
  incidents: number;
  casualties: number;
  arrests: number;
  date: string;
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ analysisData, queryResponse }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'detailed' | 'query' | 'charts'>('overview');
  const [monthlyData, setMonthlyData] = useState<MonthlyData[]>([]);
  const [availableCrimeTypes, setAvailableCrimeTypes] = useState<string[]>([]);
  const [selectedCrimeType, setSelectedCrimeType] = useState<string>('');
  const [loadingCharts, setLoadingCharts] = useState(false);
  const [fetchingCrimeTypes, setFetchingCrimeTypes] = useState(false);

  // Fetch crime types and monthly data
  useEffect(() => {
    fetchCrimeTypes();
  }, []);

  useEffect(() => {
    fetchMonthlyData();
  }, [selectedCrimeType]);

  const fetchCrimeTypes = async () => {
    try {
      setFetchingCrimeTypes(true);
      console.log('Fetching crime types for analysis results...');

      const response = await fetch('http://localhost:8000/crime-types');
      if (response.ok) {
        const data = await response.json();
        console.log('Crime types data:', data);

        setAvailableCrimeTypes(data.crime_types || []);

        // Auto-select first crime type or set to "All" mode
        if (data.crime_types && data.crime_types.length > 0) {
          setSelectedCrimeType(''); // Start with "All Types" view
        }
      }
    } catch (error) {
      console.error('Error fetching crime types:', error);
    } finally {
      setFetchingCrimeTypes(false);
    }
  };

  const fetchMonthlyData = async () => {
    try {
      setLoadingCharts(true);
      console.log('Fetching monthly chart data...');

      // Build URL with optional crime type filter
      const url = selectedCrimeType
        ? `http://localhost:8000/monthly-chart-data?crime_type=${encodeURIComponent(selectedCrimeType)}`
        : 'http://localhost:8000/monthly-chart-data';

      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        console.log('Chart data received:', data);

        if (data.status === 'success' && data.monthly_data) {
          // Data is already properly formatted from the backend
          const formattedData: MonthlyData[] = data.monthly_data.map((item: any) => ({
            month: item.month,
            incidents: item.incidents,
            casualties: item.casualties,
            arrests: item.arrests,
            date: item.date
          }));

          setMonthlyData(formattedData);
          console.log('Monthly data set:', formattedData);
        } else {
          console.warn('No monthly data available or error occurred:', data.error);
          setMonthlyData([]);
        }
      } else {
        console.error('Failed to fetch monthly data, status:', response.status);
        setMonthlyData([]);
      }
    } catch (error) {
      console.error('Error fetching monthly data:', error);
      setMonthlyData([]);
    } finally {
      setLoadingCharts(false);
    }
  };

  if (!analysisData && !queryResponse && monthlyData.length === 0) {
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

  // Enhanced chart data generation from document analysis
  const generateChartData = () => {
    if (!analysisData?.analysis?.numerical_intelligence) return [];

    const numerical = analysisData.analysis.numerical_intelligence;
    const data = [];

    const incidents = Array.isArray(numerical.incidents) ? Math.max(0, ...numerical.incidents) : 0;
    const casualties = Array.isArray(numerical.casualties) ? Math.max(0, ...numerical.casualties) : 0;
    const arrests = Array.isArray(numerical.arrests) ? Math.max(0, ...numerical.arrests) : 0;

    if (incidents > 0) data.push({ name: 'Incidents', value: incidents });
    if (casualties > 0) data.push({ name: 'Casualties', value: casualties });
    if (arrests > 0) data.push({ name: 'Arrests', value: arrests });

    return data;
  };

  const generateCrimeData = () => {
    if (!analysisData?.analysis?.crime_patterns?.primary_crimes) return [];

    const primary_crimes = analysisData.analysis.crime_patterns.primary_crimes;

    if (Array.isArray(primary_crimes)) {
      return primary_crimes.map(([crime, count]: [string, number]) => ({
        name: crime,
        value: count
      }));
    }

    if (typeof primary_crimes === 'object') {
      return Object.entries(primary_crimes).map(([crime, count]) => ({
        name: crime,
        value: typeof count === 'number' ? count : 0
      }));
    }

    return [];
  };

  // Generate summary statistics from monthly data
  const getSummaryStats = () => {
    if (!monthlyData || monthlyData.length === 0) {
      return { totalIncidents: 0, totalCasualties: 0, totalArrests: 0, avgIncidents: 0 };
    }

    const totalIncidents = monthlyData.reduce((sum, month) => sum + (month.incidents || 0), 0);
    const totalCasualties = monthlyData.reduce((sum, month) => sum + (month.casualties || 0), 0);
    const totalArrests = monthlyData.reduce((sum, month) => sum + (month.arrests || 0), 0);
    const avgIncidents = Math.round(totalIncidents / monthlyData.length);

    return { totalIncidents, totalCasualties, totalArrests, avgIncidents };
  };

  // Generate trend data for line chart
  const generateTrendData = () => {
    return monthlyData.slice(-6); // Last 6 months
  };

  // Generate distribution data for pie chart
  const generateDistributionData = () => {
    if (monthlyData.length === 0) return [];

    const stats = getSummaryStats();
    return [
      { name: 'Incidents', value: stats.totalIncidents, color: '#ef4444' },
      { name: 'Casualties', value: stats.totalCasualties, color: '#f59e0b' },
      { name: 'Arrests', value: stats.totalArrests, color: '#10b981' }
    ].filter(item => item.value > 0);
  };

  const safeGet = (obj: any, path: string, defaultValue: any = 'N/A') => {
    try {
      return path.split('.').reduce((current, key) => current?.[key], obj) ?? defaultValue;
    } catch {
      return defaultValue;
    }
  };

  const COLORS = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6'];

  const tabs = [
    { id: 'overview' as const, label: 'Overview', icon: Eye },
    { id: 'detailed' as const, label: 'Detailed Analysis', icon: Brain },
    { id: 'charts' as const, label: 'Data Charts', icon: BarChart3 },
    { id: 'query' as const, label: 'Query Results', icon: FileText }
  ];

  const summaryStats = getSummaryStats();

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Analysis Results</h2>
          <p className="text-gray-600 mt-1">
            {analysisData ? `Document: ${analysisData.metadata.filename}` : 'Intelligence Data Analysis'}
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

      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Summary Statistics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="bg-white p-6 rounded-lg shadow border">
              <p className="text-sm font-medium text-gray-600">Total Incidents</p>
              <p className="text-2xl font-bold text-red-600">{summaryStats.totalIncidents.toLocaleString()}</p>
              <p className="text-xs text-gray-500 mt-1">
                {selectedCrimeType ? `${selectedCrimeType} only` : 'All crime types'}
              </p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow border">
              <p className="text-sm font-medium text-gray-600">Total Casualties</p>
              <p className="text-2xl font-bold text-orange-600">{summaryStats.totalCasualties.toLocaleString()}</p>
              <p className="text-xs text-gray-500 mt-1">All categories</p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow border">
              <p className="text-sm font-medium text-gray-600">Total Arrests</p>
              <p className="text-2xl font-bold text-green-600">{summaryStats.totalArrests.toLocaleString()}</p>
              <p className="text-xs text-gray-500 mt-1">All categories</p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow border">
              <p className="text-sm font-medium text-gray-600">Avg Incidents/Month</p>
              <p className="text-2xl font-bold text-blue-600">{summaryStats.avgIncidents}</p>
              <p className="text-xs text-gray-500 mt-1">Monthly average</p>
            </div>
          </div>

          {/* Quick Charts Overview */}
          {monthlyData.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Trend Chart */}
              <div className="bg-white p-6 rounded-lg shadow border">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <TrendingUp className="h-5 w-5 text-blue-600 mr-2" />
                  6-Month Trend
                  {selectedCrimeType && (
                    <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                      {selectedCrimeType}
                    </span>
                  )}
                </h3>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={generateTrendData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="incidents" stroke="#ef4444" strokeWidth={2} name="Incidents" />
                    <Line type="monotone" dataKey="casualties" stroke="#f59e0b" strokeWidth={2} name="Casualties" />
                    <Line type="monotone" dataKey="arrests" stroke="#10b981" strokeWidth={2} name="Arrests" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Distribution Chart */}
              <div className="bg-white p-6 rounded-lg shadow border">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <Database className="h-5 w-5 text-purple-600 mr-2" />
                  Overall Distribution
                </h3>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={generateDistributionData()}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(1)}%`}
                    >
                      {generateDistributionData().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => value.toLocaleString()} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* AI Summary */}
          {analysisData && (
            <div className="bg-white p-6 rounded-lg shadow border">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Brain className="h-5 w-5 text-blue-600 mr-2" />
                AI Intelligence Summary
              </h3>
              <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                <p className="text-blue-800 leading-relaxed">
                  {safeGet(analysisData, 'analysis.intelligence_summary', 'No summary available.')}
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'charts' && (
        <div className="space-y-6">
          {/* Crime Type Filter - NEW */}
          <div className="bg-white p-4 rounded-lg shadow border">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900 flex items-center">
                <Filter className="h-5 w-5 text-blue-600 mr-2" />
                Chart Data Filter
              </h3>
              <div className="flex items-center gap-4">
                <select
                  value={selectedCrimeType}
                  onChange={(e) => setSelectedCrimeType(e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  disabled={fetchingCrimeTypes}
                >
                  <option value="">All Crime Types</option>
                  {availableCrimeTypes.map(type => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
                <button
                  onClick={fetchMonthlyData}
                  disabled={loadingCharts}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
                >
                  Refresh Data
                </button>
              </div>
            </div>
            {selectedCrimeType && (
              <p className="mt-2 text-sm text-gray-600">
                Showing data for: <span className="font-semibold text-blue-700">{selectedCrimeType}</span>
              </p>
            )}
            {!selectedCrimeType && (
              <p className="mt-2 text-sm text-gray-600">
                Showing aggregated data for <span className="font-semibold text-blue-700">all crime types</span>
              </p>
            )}
          </div>

          {loadingCharts ? (
            <div className="text-center p-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
              <p className="mt-2 text-gray-600">Loading filtered chart data...</p>
            </div>
          ) : monthlyData.length > 0 ? (
            <>
              {/* Monthly Trends */}
              <div className="bg-white p-6 rounded-lg shadow border">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Monthly Trends Analysis
                  {selectedCrimeType && (
                    <span className="ml-2 px-2 py-1 bg-indigo-100 text-indigo-800 text-sm rounded-full">
                      {selectedCrimeType}
                    </span>
                  )}
                </h3>
                <ResponsiveContainer width="100%" height={400}>
                  <AreaChart data={monthlyData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip formatter={(value) => value.toLocaleString()} />
                    <Legend />
                    <Area type="monotone" dataKey="incidents" stackId="1" stroke="#ef4444" fill="#ef4444" fillOpacity={0.6} name="Incidents" />
                    <Area type="monotone" dataKey="casualties" stackId="2" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.6} name="Casualties" />
                    <Area type="monotone" dataKey="arrests" stackId="3" stroke="#10b981" fill="#10b981" fillOpacity={0.6} name="Arrests" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Comparative Bar Chart */}
              <div className="bg-white p-6 rounded-lg shadow border">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Monthly Comparison
                  {selectedCrimeType && (
                    <span className="ml-2 px-2 py-1 bg-green-100 text-green-800 text-sm rounded-full">
                      {selectedCrimeType}
                    </span>
                  )}
                </h3>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={monthlyData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip formatter={(value) => value.toLocaleString()} />
                    <Legend />
                    <Bar dataKey="incidents" fill="#ef4444" name="Incidents" />
                    <Bar dataKey="casualties" fill="#f59e0b" name="Casualties" />
                    <Bar dataKey="arrests" fill="#10b981" name="Arrests" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Statistics Table */}
              <div className="bg-white p-6 rounded-lg shadow border">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Detailed Monthly Statistics
                  {selectedCrimeType && (
                    <span className="ml-2 px-2 py-1 bg-purple-100 text-purple-800 text-sm rounded-full">
                      {selectedCrimeType}
                    </span>
                  )}
                </h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Month</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Incidents</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Casualties</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Arrests</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Incident Rate</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {monthlyData.map((month, index) => (
                        <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{month.month}</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-red-600 font-semibold">{month.incidents.toLocaleString()}</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-orange-600 font-semibold">{month.casualties.toLocaleString()}</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-green-600 font-semibold">{month.arrests.toLocaleString()}</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {month.casualties > 0 ? (month.incidents / month.casualties).toFixed(2) : 'N/A'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          ) : (
            <div className="text-center p-8 bg-gray-50 border border-gray-200 rounded-lg">
              <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Chart Data Available</h3>
              <p className="text-gray-600 mb-4">
                {selectedCrimeType
                  ? `No data found for ${selectedCrimeType}. Try selecting a different crime type.`
                  : 'Upload monthly intelligence reports to see data visualizations.'
                }
              </p>
              <div className="flex justify-center gap-2">
                <button
                  onClick={fetchMonthlyData}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  Refresh Data
                </button>
                {selectedCrimeType && (
                  <button
                    onClick={() => setSelectedCrimeType('')}
                    className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
                  >
                    Show All Types
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'detailed' && analysisData && (
        <div className="space-y-6">
          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Shield className="h-5 w-5 text-blue-600 mr-2" />
              Document Classification
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-500">Primary Type:</span>
                <span className="text-gray-900 capitalize ml-2">
                  {safeGet(analysisData, 'analysis.document_classification.primary_type', 'intelligence_report').replace('_', ' ')}
                </span>
              </div>
              <div>
                <span className="font-medium text-gray-500">Security Level:</span>
                <span className="text-red-600 font-bold ml-2">
                  {safeGet(analysisData, 'analysis.document_classification.security_classification', 'RESTRICTED')}
                </span>
              </div>
              <div>
                <span className="font-medium text-gray-500">Confidence:</span>
                <span className="text-gray-900 ml-2">
                  {(safeGet(analysisData, 'analysis.document_classification.confidence', 0.8) * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Users className="h-5 w-5 text-purple-600 mr-2" />
              Extracted Entities
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Object.entries(safeGet(analysisData, 'analysis.entities', {})).map(([type, list]) => (
                <div key={type}>
                  <h4 className="font-medium text-gray-800 capitalize mb-2 border-b pb-1">{type}</h4>
                  {Array.isArray(list) && list.length > 0 ? (
                    <ul className="space-y-1 text-sm text-gray-600">
                      {list.slice(0, 5).map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                      {list.length > 5 && (
                        <li className="text-xs text-gray-400">...and {list.length - 5} more.</li>
                      )}
                    </ul>
                  ) : (
                    <p className="text-sm text-gray-400">None identified.</p>
                  )}
                </div>
              ))}
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