// src/components/ForecastingDashboard.tsx - AI-Powered Forecasting Dashboard (OPERATIONAL)

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { TrendingUp, TrendingDown, Brain, AlertTriangle, Target, Activity, Zap } from 'lucide-react';

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
      <div className="p-8">
        <div className="text-center">
          <div className="inline-flex items-center px-4 py-2 bg-blue-50 border border-blue-200 rounded-lg">
            <Brain className="animate-pulse h-5 w-5 text-blue-600 mr-2" />
            <span className="text-blue-700 font-medium">Loading historical data and generating live forecast...</span>
          </div>
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
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Retry Forecasting
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
            Live predictions based on all uploaded intelligence reports.
          </p>
        </div>
      </div>

      {threatMetrics && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white p-6 rounded-lg shadow border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Current Threat Level</p>
                <p className={`text-2xl font-bold ${
                  threatMetrics.current_threat_level >= 70 ? 'text-red-600' :
                  threatMetrics.current_threat_level >= 50 ? 'text-orange-600' : 'text-green-600'
                }`}>
                  {threatMetrics.current_threat_level.toFixed(1)}%
                </p>
              </div>
              <Target className={`h-8 w-8 ${
                threatMetrics.current_threat_level >= 70 ? 'text-red-500' :
                threatMetrics.current_threat_level >= 50 ? 'text-orange-500' : 'text-green-500'
              }`} />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Change from Prior Month</p>
                <p className={`text-2xl font-bold flex items-center ${threatMetrics.predicted_change > 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {threatMetrics.predicted_change > 0 ? '+' : ''}{threatMetrics.predicted_change.toFixed(1)}%
                  {getTrendIcon(threatMetrics.predicted_change)}
                </p>
              </div>
              <Activity className="h-8 w-8 text-blue-500" />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Model Confidence</p>
                <p className="text-2xl font-bold text-green-600">
                  {threatMetrics.confidence_score.toFixed(1)}%
                </p>
              </div>
              <Brain className="h-8 w-8 text-green-500" />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Identified Risk Factors</p>
                <p className="text-2xl font-bold text-gray-600">
                  {threatMetrics.risk_factors.length}
                </p>
              </div>
              <Zap className="h-8 w-8 text-gray-500" />
            </div>
          </div>
        </div>
      )}

      <div className="bg-white p-6 rounded-lg shadow border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <TrendingUp className="h-5 w-5 text-blue-600 mr-2" />
          Incident Forecast Model
        </h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={forecastData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' })}
              />
              <YAxis domain={['dataMin - 50', 'dataMax + 50']} />
              <Tooltip
                labelFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'long', year: 'numeric', timeZone: 'UTC' })}
                formatter={(value: number, name: string) => [value, name === 'incidents' ? 'Actual Incidents' : 'Forecasted Incidents']}
              />
              <Legend />
              <Line connectNulls type="monotone" dataKey="incidents" stroke="#3b82f6" strokeWidth={2} name="Actual Incidents" />
              <Line connectNulls type="monotone" dataKey="predicted_incidents" stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" name="Forecasted Incidents" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {threatMetrics && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <AlertTriangle className="h-5 w-5 text-red-600 mr-2" />
              Key Risk Factors (from Reports)
            </h3>
            <div className="space-y-3">
              {threatMetrics.risk_factors.map((factor, index) => (
                <div key={index} className="flex items-start p-3 bg-red-50 border border-red-200 rounded-lg">
                  <AlertTriangle className="h-4 w-4 text-red-600 mr-3 flex-shrink-0 mt-1" />
                  <span className="text-red-800 text-sm">{factor}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Brain className="h-5 w-5 text-green-600 mr-2" />
              Strategic Recommendations (from Reports)
            </h3>
            <div className="space-y-3">
              {threatMetrics.recommendations.map((recommendation, index) => (
                <div key={index} className="flex items-start p-3 bg-green-50 border border-green-200 rounded-lg">
                  <Target className="h-4 w-4 text-green-600 mr-3 flex-shrink-0 mt-1" />
                  <span className="text-green-800 text-sm">{recommendation}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ForecastingDashboard;