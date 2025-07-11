// src/components/ForecastingDashboard.tsx - Enhanced with Multi-Crime Type Support

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { TrendingUp, TrendingDown, Brain, AlertTriangle, Target, Activity, Zap, Loader2, RefreshCw, BarChart3, Filter } from 'lucide-react';

interface ForecastingData {
  date: string;
  incidents: number | null;
  predicted_incidents: number | null;
}

interface ThreatMetrics {
  current_threat_level: number;
  predicted_change: number;
  confidence_score: number;
  risk_factors: string[];
  recommendations: string[];
}

const ForecastingDashboard: React.FC = () => {
  const [forecastData, setForecastData] = useState<ForecastingData[]>([]);
  const [threatMetrics, setThreatMetrics] = useState<ThreatMetrics | null>(null);
  const [availableCrimeTypes, setAvailableCrimeTypes] = useState<string[]>([]);
  const [selectedCrimeType, setSelectedCrimeType] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [fetchingCrimeTypes, setFetchingCrimeTypes] = useState(true);

  useEffect(() => {
    fetchCrimeTypes();
  }, []);

  useEffect(() => {
    if (selectedCrimeType) {
      fetchForecastingData();
    }
  }, [selectedCrimeType]);

  const fetchCrimeTypes = async () => {
    try {
      setFetchingCrimeTypes(true);
      setError(null);

      console.log('Fetching crime types for forecasting...');
      const response = await fetch('http://localhost:8000/crime-types');

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Crime types for forecasting:', data);

      const crimeTypes = data.crime_types || [];
      setAvailableCrimeTypes(crimeTypes);

      if (crimeTypes.length > 0) {
        setSelectedCrimeType(crimeTypes[0]); // Auto-select first crime type
        console.log('Auto-selected crime type for forecasting:', crimeTypes[0]);
      } else {
        setError("No crime types available for forecasting. Please upload some documents first.");
        setLoading(false);
      }

    } catch (err) {
      console.error('Failed to fetch crime types:', err);
      setError(`Could not load crime types: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setAvailableCrimeTypes([]);
      setLoading(false);
    } finally {
      setFetchingCrimeTypes(false);
    }
  };

  const fetchForecastingData = async () => {
    if (!selectedCrimeType) return;

    try {
      setLoading(true);
      setError(null);

      console.log(`Fetching forecasting data for: ${selectedCrimeType}`);
      const url = `http://localhost:8000/forecast?crime_type=${encodeURIComponent(selectedCrimeType)}`;
      const response = await fetch(url);

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: `HTTP error! status: ${response.status}` }));
        throw new Error(errData.detail);
      }

      const data = await response.json();
      console.log('Forecasting data received:', data);

      setForecastData(data.forecastData || []);
      setThreatMetrics(data.threatMetrics || null);

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch forecasting data';
      console.error('Forecasting error:', errorMessage);
      setError(errorMessage);
      setForecastData([]);
      setThreatMetrics(null);
    } finally {
      setLoading(false);
    }
  };

  const handleCrimeTypeChange = (newCrimeType: string) => {
    setSelectedCrimeType(newCrimeType);
    // Clear previous data immediately to show loading state
    setForecastData([]);
    setThreatMetrics(null);
  };

  const getTrendIcon = (change: number) => {
    if (change > 0) return <TrendingUp className="h-4 w-4 text-red-500 ml-1" />;
    if (change < 0) return <TrendingDown className="h-4 w-4 text-green-500 ml-1" />;
    return <Activity className="h-4 w-4 text-gray-500 ml-1" />;
  };

  const getThreatLevelColor = (level: number) => {
    if (level >= 70) return 'text-red-600';
    if (level >= 50) return 'text-orange-600';
    return 'text-green-600';
  };

  const getThreatLevelBg = (level: number) => {
    if (level >= 70) return 'bg-red-50 border-red-200';
    if (level >= 50) return 'bg-orange-50 border-orange-200';
    return 'bg-green-50 border-green-200';
  };

  if (loading && fetchingCrimeTypes) {
    return (
      <div className="p-8 flex justify-center items-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 mx-auto animate-spin text-blue-600" />
          <p className="mt-2 text-gray-600">Initializing forecasting system...</p>
        </div>
      </div>
    );
  }

  if (error && !selectedCrimeType) {
    return (
      <div className="p-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 text-red-600 mr-2" />
            <h3 className="text-lg font-medium text-red-800">Forecasting Error</h3>
          </div>
          <p className="text-red-700 mt-2">{error}</p>
          <button
            onClick={fetchCrimeTypes}
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 flex items-center">
            <Brain className="h-6 w-6 text-purple-600 mr-2" />
            AI Forecasting Dashboard
          </h2>
          <p className="text-gray-600 mt-1">
            Crime-specific predictions based on historical intelligence data using ARIMA modeling.
          </p>
        </div>
      </div>

      {/* Crime Type Selector */}
      <div className="bg-white p-6 rounded-lg shadow border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Target className="h-5 w-5 text-blue-600 mr-2" />
          Select Crime Type for Forecasting
        </h3>
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <label htmlFor="crimeTypeSelect" className="block text-sm font-medium text-gray-700 mb-2">
              Crime Category
              {fetchingCrimeTypes && <Loader2 className="inline h-4 w-4 ml-2 animate-spin" />}
            </label>
            <select
              id="crimeTypeSelect"
              value={selectedCrimeType}
              onChange={(e) => handleCrimeTypeChange(e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              disabled={fetchingCrimeTypes || availableCrimeTypes.length === 0}
            >
              <option value="" disabled>
                {fetchingCrimeTypes ? 'Loading crime types...' : 'Choose a crime type to forecast...'}
              </option>
              {availableCrimeTypes.map(type => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>

          <div className="flex gap-2">
            <button
              onClick={fetchCrimeTypes}
              disabled={fetchingCrimeTypes}
              className="px-4 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:bg-gray-100 mt-6"
              title="Refresh crime types"
            >
              <RefreshCw className={`h-5 w-5 ${fetchingCrimeTypes ? 'animate-spin' : ''}`} />
            </button>

            <button
              onClick={fetchForecastingData}
              disabled={!selectedCrimeType || loading}
              className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400 mt-6 flex items-center"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Zap className="h-4 w-4 mr-2" />}
              {loading ? 'Generating...' : 'Update Forecast'}
            </button>
          </div>
        </div>

        {selectedCrimeType && (
          <div className="mt-4 p-4 bg-purple-50 border border-purple-200 rounded-lg">
            <p className="text-sm text-purple-800">
              <span className="font-medium">Forecasting:</span> {selectedCrimeType}
            </p>
            <p className="text-xs text-purple-600 mt-1">
              Historical data analysis and 6-month predictive modeling
            </p>
          </div>
        )}
      </div>

      {/* Loading State for Forecast Data */}
      {loading && selectedCrimeType && (
        <div className="bg-white p-8 rounded-lg shadow border">
          <div className="text-center">
            <Loader2 className="h-8 w-8 mx-auto animate-spin text-purple-600" />
            <p className="mt-2 text-gray-600">Generating forecast for {selectedCrimeType}...</p>
            <p className="text-sm text-gray-500 mt-1">Analyzing historical patterns and trends</p>
          </div>
        </div>
      )}

      {/* Error State for Forecast Data */}
      {error && selectedCrimeType && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 text-red-600 mr-2" />
            <h3 className="text-lg font-medium text-red-800">Forecasting Error</h3>
          </div>
          <p className="text-red-700 mt-2">{error}</p>
          <p className="text-sm text-red-600 mt-1">
            This may occur if there's insufficient historical data for {selectedCrimeType}.
          </p>
          <div className="mt-4 flex gap-2">
            <button
              onClick={fetchForecastingData}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
            >
              Retry Forecasting
            </button>
            <button
              onClick={() => setSelectedCrimeType('')}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
            >
              Select Different Type
            </button>
          </div>
        </div>
      )}

      {/* Main Dashboard Content */}
      {!loading && !error && selectedCrimeType && threatMetrics && (
        <>
          {/* Threat Metrics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className={`bg-white p-6 rounded-lg shadow border ${getThreatLevelBg(threatMetrics.current_threat_level)}`}>
              <p className="text-sm font-medium text-gray-600">Current Threat Level</p>
              <p className={`text-2xl font-bold ${getThreatLevelColor(threatMetrics.current_threat_level)}`}>
                {threatMetrics.current_threat_level?.toFixed(1) ?? '0.0'}%
              </p>
              <p className="text-xs text-gray-500 mt-1">{selectedCrimeType}</p>
            </div>

            <div className="bg-white p-6 rounded-lg shadow border">
              <p className="text-sm font-medium text-gray-600">Predicted Change</p>
              <p className={`text-2xl font-bold flex items-center ${(threatMetrics.predicted_change ?? 0) > 0 ? 'text-red-600' : 'text-green-600'}`}>
                {(threatMetrics.predicted_change ?? 0) > 0 ? '+' : ''}{threatMetrics.predicted_change?.toFixed(1) ?? '0.0'}%
                {getTrendIcon(threatMetrics.predicted_change ?? 0)}
              </p>
              <p className="text-xs text-gray-500 mt-1">Next month prediction</p>
            </div>

            <div className="bg-white p-6 rounded-lg shadow border">
              <p className="text-sm font-medium text-gray-600">Model Confidence</p>
              <p className="text-2xl font-bold text-green-600">
                {threatMetrics.confidence_score?.toFixed(1) ?? '0.0'}%
              </p>
              <p className="text-xs text-gray-500 mt-1">ARIMA model accuracy</p>
            </div>

            <div className="bg-white p-6 rounded-lg shadow border">
              <p className="text-sm font-medium text-gray-600">Risk Factors</p>
              <p className="text-2xl font-bold text-orange-600">
                {threatMetrics.risk_factors?.length ?? 0}
              </p>
              <p className="text-xs text-gray-500 mt-1">Identified indicators</p>
            </div>
          </div>

          {/* Forecast Chart */}
          {forecastData.length > 0 && (
            <div className="bg-white p-6 rounded-lg shadow border">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <BarChart3 className="h-5 w-5 text-blue-600 mr-2" />
                {selectedCrimeType} - Incident Forecast Model
                <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                  6-Month Prediction
                </span>
              </h3>
              <div style={{ height: '400px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={forecastData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="date"
                      tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', year: '2-digit', timeZone: 'UTC' })}
                    />
                    <YAxis domain={['dataMin - 10', 'dataMax + 50']} />
                    <Tooltip
                      labelFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'long', year: 'numeric', timeZone: 'UTC' })}
                      formatter={(value, name) => [
                        value ? value.toLocaleString() : 'N/A',
                        name === 'incidents' ? 'Historical Data' : 'AI Forecast'
                      ]}
                    />
                    <Legend />
                    <Area
                      connectNulls={false}
                      type="monotone"
                      dataKey="incidents"
                      stroke="#3b82f6"
                      fill="#3b82f6"
                      fillOpacity={0.3}
                      strokeWidth={2}
                      name="Historical Incidents"
                    />
                    <Area
                      connectNulls={false}
                      type="monotone"
                      dataKey="predicted_incidents"
                      stroke="#ef4444"
                      fill="#ef4444"
                      fillOpacity={0.2}
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      name="Predicted Incidents"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-600">
                  <strong>Model:</strong> ARIMA time series analysis trained on historical {selectedCrimeType.toLowerCase()} incident data.
                  Dashed lines represent AI predictions with {threatMetrics.confidence_score?.toFixed(1)}% confidence.
                </p>
              </div>
            </div>
          )}

          {/* Risk Analysis */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white p-6 rounded-lg shadow border">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <AlertTriangle className="h-5 w-5 text-red-600 mr-2" />
                Key Risk Factors
              </h3>
              {threatMetrics.risk_factors && threatMetrics.risk_factors.length > 0 ? (
                <ul className="space-y-3">
                  {threatMetrics.risk_factors.map((factor, index) => (
                    <li key={index} className="flex items-start text-sm">
                      <AlertTriangle className="h-4 w-4 text-red-500 mr-3 mt-0.5 flex-shrink-0" />
                      <span className="text-gray-700">{factor}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-gray-500 text-sm">No specific risk factors identified for {selectedCrimeType}.</p>
              )}
            </div>

            <div className="bg-white p-6 rounded-lg shadow border">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Target className="h-5 w-5 text-green-600 mr-2" />
                Strategic Recommendations
              </h3>
              {threatMetrics.recommendations && threatMetrics.recommendations.length > 0 ? (
                <ul className="space-y-3">
                  {threatMetrics.recommendations.map((rec, index) => (
                    <li key={index} className="flex items-start text-sm">
                      <Target className="h-4 w-4 text-green-500 mr-3 mt-0.5 flex-shrink-0" />
                      <span className="text-gray-700">{rec}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="text-sm text-gray-500">
                  <p className="mb-2">Standard recommendations for {selectedCrimeType}:</p>
                  <ul className="space-y-2">
                    <li className="flex items-start">
                      <Target className="h-4 w-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                      Monitor trend patterns and deploy resources accordingly
                    </li>
                    <li className="flex items-start">
                      <Target className="h-4 w-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                      Enhance intelligence gathering in high-risk areas
                    </li>
                    <li className="flex items-start">
                      <Target className="h-4 w-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                      Coordinate with relevant security agencies
                    </li>
                  </ul>
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {/* Initial State - No Crime Type Selected */}
      {!selectedCrimeType && !fetchingCrimeTypes && availableCrimeTypes.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-8 text-center">
          <Brain className="h-12 w-12 text-blue-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-blue-800 mb-2">Select Crime Type to Begin Forecasting</h3>
          <p className="text-blue-700 mb-4">
            Choose from {availableCrimeTypes.length} available crime categories to generate AI-powered predictions and trend analysis.
          </p>
          <div className="flex flex-wrap justify-center gap-2">
            {availableCrimeTypes.slice(0, 3).map(type => (
              <button
                key={type}
                onClick={() => setSelectedCrimeType(type)}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm"
              >
                Forecast {type}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ForecastingDashboard;