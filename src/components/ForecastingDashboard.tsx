// src/components/ForecastingDashboard.tsx - AI-Powered Forecasting Dashboard

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, AreaChart, Area } from 'recharts';
import { TrendingUp, TrendingDown, Brain, AlertTriangle, Calendar, Target, Activity, Zap } from 'lucide-react';

interface ForecastingData {
  date: string;
  incidents: number;
  predicted_incidents: number;
  threat_level: number;
  confidence: number;
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
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d'>('30d');

  useEffect(() => {
    fetchForecastingData();
  }, [timeRange]);

  const fetchForecastingData = async () => {
    try {
      setLoading(true);
      // Simulate API call - replace with actual endpoint
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Generate mock data for demonstration
      const mockData: ForecastingData[] = [];
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - parseInt(timeRange.replace('d', '')));

      for (let i = 0; i < parseInt(timeRange.replace('d', '')); i++) {
        const date = new Date(startDate);
        date.setDate(date.getDate() + i);

        mockData.push({
          date: date.toISOString().split('T')[0],
          incidents: Math.floor(Math.random() * 20) + 5,
          predicted_incidents: Math.floor(Math.random() * 25) + 3,
          threat_level: Math.random() * 100,
          confidence: Math.random() * 40 + 60
        });
      }

      setForecastData(mockData);

      setThreatMetrics({
        current_threat_level: 67,
        predicted_change: 12,
        confidence_score: 85,
        risk_factors: [
          'Increased regional tensions',
          'Seasonal criminal activity patterns',
          'Economic instability indicators'
        ],
        recommendations: [
          'Enhanced surveillance in high-risk areas',
          'Increased patrol frequency during peak hours',
          'Coordinate with local intelligence units'
        ]
      });

      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch forecasting data');
    } finally {
      setLoading(false);
    }
  };

  const getThreatLevelColor = (level: number) => {
    if (level >= 80) return 'text-red-600 bg-red-50 border-red-200';
    if (level >= 60) return 'text-orange-600 bg-orange-50 border-orange-200';
    if (level >= 40) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-green-600 bg-green-50 border-green-200';
  };

  const getTrendIcon = (change: number) => {
    if (change > 0) return <TrendingUp className="h-4 w-4 text-red-500" />;
    if (change < 0) return <TrendingDown className="h-4 w-4 text-green-500" />;
    return <Activity className="h-4 w-4 text-gray-500" />;
  };

  if (loading) {
    return (
      <div className="p-8">
        <div className="text-center">
          <div className="inline-flex items-center px-4 py-2 bg-blue-50 border border-blue-200 rounded-lg">
            <Brain className="animate-pulse h-5 w-5 text-blue-600 mr-2" />
            <span className="text-blue-700 font-medium">Generating AI forecasts...</span>
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
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 flex items-center">
            <Brain className="h-6 w-6 text-purple-600 mr-2" />
            AI Forecasting Dashboard
          </h2>
          <p className="text-gray-600 mt-1">
            Machine learning predictions and threat analysis
          </p>
        </div>

        <div className="flex items-center space-x-4">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as any)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last 90 Days</option>
          </select>
        </div>
      </div>

      {/* Key Metrics */}
      {threatMetrics && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white p-6 rounded-lg shadow border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Current Threat Level</p>
                <p className={`text-2xl font-bold ${
                  threatMetrics.current_threat_level >= 80 ? 'text-red-600' :
                  threatMetrics.current_threat_level >= 60 ? 'text-orange-600' :
                  threatMetrics.current_threat_level >= 40 ? 'text-yellow-600' : 'text-green-600'
                }`}>
                  {threatMetrics.current_threat_level}%
                </p>
              </div>
              <Target className={`h-8 w-8 ${
                threatMetrics.current_threat_level >= 80 ? 'text-red-500' :
                threatMetrics.current_threat_level >= 60 ? 'text-orange-500' :
                threatMetrics.current_threat_level >= 40 ? 'text-yellow-500' : 'text-green-500'
              }`} />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Predicted Change</p>
                <p className="text-2xl font-bold text-blue-600 flex items-center">
                  {threatMetrics.predicted_change > 0 ? '+' : ''}{threatMetrics.predicted_change}%
                  {getTrendIcon(threatMetrics.predicted_change)}
                </p>
              </div>
              <Activity className="h-8 w-8 text-blue-500" />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">AI Confidence</p>
                <p className="text-2xl font-bold text-green-600">
                  {threatMetrics.confidence_score}%
                </p>
              </div>
              <Brain className="h-8 w-8 text-green-500" />
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow border">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Risk Factors</p>
                <p className="text-2xl font-bold text-gray-600">
                  {threatMetrics.risk_factors.length}
                </p>
              </div>
              <Zap className="h-8 w-8 text-gray-500" />
            </div>
          </div>
        </div>
      )}

      {/* Forecast Chart */}
      <div className="bg-white p-6 rounded-lg shadow border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <TrendingUp className="h-5 w-5 text-blue-600 mr-2" />
          Incident Prediction Model
        </h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={forecastData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <YAxis />
              <Tooltip
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
                formatter={(value: number, name: string) => [value, name === 'incidents' ? 'Actual Incidents' : 'Predicted Incidents']}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="incidents"
                stroke="#3b82f6"
                strokeWidth={2}
                name="Actual Incidents"
              />
              <Line
                type="monotone"
                dataKey="predicted_incidents"
                stroke="#ef4444"
                strokeWidth={2}
                strokeDasharray="5 5"
                name="Predicted Incidents"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Threat Level Trend */}
      <div className="bg-white p-6 rounded-lg shadow border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <AlertTriangle className="h-5 w-5 text-orange-600 mr-2" />
          Threat Level Analysis
        </h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={forecastData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <YAxis domain={[0, 100]} />
              <Tooltip
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
                formatter={(value: number) => [`${value.toFixed(1)}%`, 'Threat Level']}
              />
              <Area
                type="monotone"
                dataKey="threat_level"
                stroke="#f59e0b"
                fill="#fef3c7"
                name="Threat Level"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Risk Analysis */}
      {threatMetrics && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Risk Factors */}
          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <AlertTriangle className="h-5 w-5 text-red-600 mr-2" />
              Identified Risk Factors
            </h3>
            <div className="space-y-3">
              {threatMetrics.risk_factors.map((factor, index) => (
                <div key={index} className="flex items-center p-3 bg-red-50 border border-red-200 rounded-lg">
                  <AlertTriangle className="h-4 w-4 text-red-600 mr-3 flex-shrink-0" />
                  <span className="text-red-800">{factor}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Recommendations */}
          <div className="bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Brain className="h-5 w-5 text-green-600 mr-2" />
              AI Recommendations
            </h3>
            <div className="space-y-3">
              {threatMetrics.recommendations.map((recommendation, index) => (
                <div key={index} className="flex items-center p-3 bg-green-50 border border-green-200 rounded-lg">
                  <Target className="h-4 w-4 text-green-600 mr-3 flex-shrink-0" />
                  <span className="text-green-800">{recommendation}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Model Performance */}
      <div className="bg-white p-6 rounded-lg shadow border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Brain className="h-5 w-5 text-purple-600 mr-2" />
          Model Confidence Over Time
        </h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={forecastData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <YAxis domain={[0, 100]} />
              <Tooltip
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
                formatter={(value: number) => [`${value.toFixed(1)}%`, 'Confidence']}
              />
              <Bar
                dataKey="confidence"
                fill="#8b5cf6"
                name="Model Confidence"
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default ForecastingDashboard;