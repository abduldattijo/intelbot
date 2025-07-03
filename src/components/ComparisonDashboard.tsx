// src/components/ComparisonDashboard.tsx - New Feature

import React, { useState, useEffect } from 'react';
import { BarChart, ArrowRight, Brain, AlertTriangle, Loader2 } from 'lucide-react';

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

const ComparisonDashboard: React.FC = () => {
  const [availableMonths, setAvailableMonths] = useState<string[]>([]);
  const [month1, setMonth1] = useState<string>('');
  const [month2, setMonth2] = useState<string>('');
  const [comparisonData, setComparisonData] = useState<ComparisonResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAvailableMonths();
  }, []);

  const fetchAvailableMonths = async () => {
    try {
      const response = await fetch('http://localhost:8000/available-months');
      if (!response.ok) throw new Error('Failed to fetch available months');

      const data = await response.json();
      const months = data.available_months || [];
      setAvailableMonths(months);

      if (months.length >= 2) {
        setMonth1(months[0]); // Most recent
        setMonth2(months[1]); // Second most recent
      }
    } catch (err) {
      console.error(err);
      setError("Could not load list of available months from the server.");
    }
  };

  const handleCompare = async () => {
    if (!month1 || !month2) {
      setError('Please select two different months to compare.');
      return;
    }
    setLoading(true);
    setError(null);
    setComparisonData(null);

    try {
      const response = await fetch(`http://localhost:8000/compare-months?month1=${month1}&month2=${month2}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'}
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to fetch comparison data.');
      }
      const data: ComparisonResponse = await response.json();
      setComparisonData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const getChangeColor = (change: string) => {
    if (change.startsWith('+')) return 'text-red-600 bg-red-50';
    if (change.startsWith('-')) return 'text-green-600 bg-green-50';
    return 'text-gray-500 bg-gray-50';
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 flex items-center">
          <BarChart className="h-6 w-6 text-indigo-600 mr-2" />
          Monthly Intelligence Comparison
        </h2>
        <p className="text-gray-600 mt-1">
          Select two months to compare key metrics and generate an AI-powered strategic inference.
        </p>
      </div>

      <div className="bg-white p-4 rounded-lg shadow border flex flex-col md:flex-row items-center gap-4">
        <div className="flex-1 w-full">
          <label htmlFor="month2" className="block text-sm font-medium text-gray-700">Older Month (Baseline)</label>
          <select id="month2" value={month2} onChange={(e) => setMonth2(e.target.value)}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md">
            <option value="" disabled>Select Month</option>
            {availableMonths.map(m => (<option key={m} value={m}>{new Date(m).toLocaleString('default', { month: 'long', year: 'numeric' })}</option>))}
          </select>
        </div>

        <ArrowRight className="h-6 w-6 text-gray-400 mt-6 hidden md:block" />

        <div className="flex-1 w-full">
           <label htmlFor="month1" className="block text-sm font-medium text-gray-700">Newer Month</label>
          <select id="month1" value={month1} onChange={(e) => setMonth1(e.target.value)}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md">
            <option value="" disabled>Select Month</option>
             {availableMonths.map(m => (<option key={m} value={m}>{new Date(m).toLocaleString('default', { month: 'long', year: 'numeric' })}</option>))}
          </select>
        </div>

        <button onClick={handleCompare} disabled={loading || !month1 || !month2}
          className="w-full md:w-auto mt-4 md:mt-0 px-6 py-3 bg-indigo-600 text-white font-semibold rounded-lg shadow-md hover:bg-indigo-700 disabled:bg-gray-400 transition-colors flex items-center justify-center">
          {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : 'Compare'}
        </button>
      </div>

       {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 text-red-600 mr-2" />
            <h3 className="text-red-800 font-medium">Comparison Error</h3>
          </div>
          <p className="text-red-700 mt-1">{error}</p>
        </div>
      )}

      {loading && (
          <div className="text-center p-8">
              <Loader2 className="h-8 w-8 text-indigo-600 mx-auto animate-spin" />
              <p className="mt-2 text-gray-600">Generating comparison and AI inference...</p>
          </div>
      )}

      {comparisonData && (
        <div className="space-y-6">
            <div className="bg-white rounded-lg shadow border p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Quantitative Comparison</h3>
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
                                    <td className={`px-6 py-4 whitespace-nowrap text-sm font-bold`}><span className={`px-2 py-1 rounded-md ${getChangeColor(row.change)}`}>{row.change}</span></td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            <div className="bg-white rounded-lg shadow border p-6">
                 <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                    <Brain className="h-5 w-5 text-indigo-600 mr-2" />
                    AI-Powered Strategic Inference
                </h3>
                <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 prose prose-sm max-w-none">
                    <p className="text-indigo-900 whitespace-pre-wrap">{comparisonData.ai_inference}</p>
                </div>
            </div>
        </div>
      )}
    </div>
  );
};

export default ComparisonDashboard;