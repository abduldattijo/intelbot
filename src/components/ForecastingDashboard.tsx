// src/components/ForecastingDashboard.tsx - AI-Powered Forecasting Dashboard (FIXED)

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { TrendingUp, TrendingDown, Brain, AlertTriangle, Target, Activity, Zap, Loader2 } from 'lucide-react';

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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchForecastingData();
  }, []);

  const fetchForecastingData = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('http://localhost:8000/forecast');
      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: `HTTP error! status: ${response.status}` }));
        throw new Error(errData.detail);
      }
      const data = await response.json();

      setForecastData(data.forecastData || []);
      setThreatMetrics(data.threatMetrics || null);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch forecasting data');
      setForecastData([]);
      setThreatMetrics(null);
    } finally {
      setLoading(false);
    }
  };

  const getTrendIcon = (change: number) => {
    if (change > 0) return <TrendingUp className="h-4 w-4 text-red-500 ml-1" />;
    if (change < 0) return <TrendingDown className="h-4 w-4 text-green-500 ml-1" />;
    return <Activity className="h-4 w-4 text-gray-500 ml-1" />;
  };

  if (loading) {
    return (
      <div className="p-8 flex justify-center items-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 mx-auto animate-spin text-blue-600" />
          <p className="mt-2 text-gray-600">Loading historical data and generating live forecast...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 text-red-600 mr-2" />
            <h3 className="text-lg font-medium text-red-800">Forecasting Error</h3>
          </div>
          <p className="text-red-700 mt-2">{error}</p>
          <button
            onClick={fetchForecastingData}
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            Retry Forecasting
          </button>
        </div>
      </div>
    );
  }

  // <<< FIX: Use Optional Chaining (?.) and Nullish Coalescing (??) for safety >>>
  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 flex items-center">
            <Brain className="h-6 w-6 text-purple-600 mr-2" />
            AI Forecasting Dashboard
          </h2>
          <p className="text-gray-600 mt-1">
            Live predictions based on all uploaded intelligence reports.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg shadow border">
          <p className="text-sm font-medium text-gray-600">Current Threat Level</p>
          <p className={`text-2xl font-bold ${ (threatMetrics?.current_threat_level ?? 0) >= 70 ? 'text-red-600' : (threatMetrics?.current_threat_level ?? 0) >= 50 ? 'text-orange-600' : 'text-green-600'}`}>
            {threatMetrics?.current_threat_level?.toFixed(1) ?? '0.0'}%
          </p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow border">
          <p className="text-sm font-medium text-gray-600">Change from Prior Month</p>
          <p className={`text-2xl font-bold flex items-center ${(threatMetrics?.predicted_change ?? 0) > 0 ? 'text-red-600' : 'text-green-600'}`}>
            {(threatMetrics?.predicted_change ?? 0) > 0 ? '+' : ''}{threatMetrics?.predicted_change?.toFixed(1) ?? '0.0'}%
            {getTrendIcon(threatMetrics?.predicted_change ?? 0)}
          </p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow border">
          <p className="text-sm font-medium text-gray-600">Model Confidence</p>
          <p className="text-2xl font-bold text-green-600">
            {threatMetrics?.confidence_score?.toFixed(1) ?? '0.0'}%
          </p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow border">
            <p className="text-sm font-medium text-gray-600">Identified Risk Factors</p>
            <p className="text-2xl font-bold text-gray-600">
              {threatMetrics?.risk_factors?.length ?? 0}
            </p>
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg shadow border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <TrendingUp className="h-5 w-5 text-blue-600 mr-2" />
          Incident Forecast Model
        </h3>
        <div style={{ height: '320px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={forecastData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' })} />
              <YAxis domain={['dataMin - 50', 'dataMax + 50']} />
              <Tooltip labelFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'long', year: 'numeric', timeZone: 'UTC' })} />
              <Legend />
              <Line connectNulls type="monotone" dataKey="incidents" stroke="#3b82f6" strokeWidth={2} name="Actual Incidents" />
              <Line connectNulls type="monotone" dataKey="predicted_incidents" stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" name="Forecasted Incidents" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Risk Factors</h3>
          <ul className="space-y-2">
            {(threatMetrics?.risk_factors ?? ['No data']).map((factor, index) => (
              <li key={index} className="flex items-start text-sm">
                <AlertTriangle className="h-4 w-4 text-red-500 mr-2 mt-0.5 flex-shrink-0" />
                <span>{factor}</span>
              </li>
            ))}
          </ul>
        </div>
        <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Strategic Recommendations</h3>
            <ul className="space-y-2">
            {(threatMetrics?.recommendations ?? ['No data']).map((rec, index) => (
              <li key={index} className="flex items-start text-sm">
                <Target className="h-4 w-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ForecastingDashboard;