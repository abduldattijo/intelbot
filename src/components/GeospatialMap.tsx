// src/components/GeospatialMap.tsx - Enhanced Multi-Crime Nigerian States Intelligence Map with Full Crime Type Filtering

import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L, { LatLngExpression } from 'leaflet';

import { IntelligenceDocument } from '../App';
import { MapPin, AlertTriangle, RefreshCw, BarChart3, Users, TrendingUp, Filter, Target, Database, Shield } from 'lucide-react';

interface GeospatialPoint {
  id: string;
  latitude: number;
  longitude: number;
  title: string;
  threat_level: 'low' | 'medium' | 'high';
}

interface NigerianStateData {
  id: string;
  name: string;
  capital: string;
  latitude: number;
  longitude: number;
  incidents: number;
  mentions: number;
  threat_level: 'low' | 'medium' | 'high';
}

interface CrimeTypeOption {
  name: string;
  count: number;
}

interface GeospatialMapProps {
  analysisData?: IntelligenceDocument | null;
}

// Helper component to automatically adjust map bounds
const MapBoundsUpdater: React.FC<{ points: GeospatialPoint[], statesData: NigerianStateData[] }> = ({ points, statesData }) => {
  const map = useMap();
  useEffect(() => {
    const allPoints = [
      ...points.map(p => [p.latitude, p.longitude] as LatLngExpression),
      ...statesData.map(s => [s.latitude, s.longitude] as LatLngExpression)
    ];

    if (allPoints.length > 0) {
      const bounds = L.latLngBounds(allPoints);
      map.fitBounds(bounds, { padding: [20, 20] });
    } else {
      map.setView([9.0820, 8.6753], 6);
    }
  }, [points, statesData, map]);
  return null;
};

const GeospatialMap: React.FC<GeospatialMapProps> = ({ analysisData }) => {
  const [points, setPoints] = useState<GeospatialPoint[]>([]);
  const [statesData, setStatesData] = useState<NigerianStateData[]>([]);
  const [availableCrimeTypes, setAvailableCrimeTypes] = useState<CrimeTypeOption[]>([]);
  const [selectedCrimeTypeFilter, setSelectedCrimeTypeFilter] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showStates, setShowStates] = useState(true);
  const [showDocumentPoints, setShowDocumentPoints] = useState(true);
  const [viewMode, setViewMode] = useState<'all' | 'high_threat' | 'recent'>('all');

  const nigeriaCenter: LatLngExpression = [9.0820, 8.6753];

  useEffect(() => {
    fetchCrimeTypes();
  }, []);

  // NEW: Refresh states data when crime type filter changes
  useEffect(() => {
    fetchNigerianStatesData();
  }, [selectedCrimeTypeFilter]);

  useEffect(() => {
    if (analysisData?.analysis?.geographic_intelligence?.coordinates) {
      const backendCoordinates = analysisData.analysis.geographic_intelligence.coordinates;

      const newPoints: GeospatialPoint[] = backendCoordinates.map((coord, index) => ({
        id: `${coord.location_name}-${index}`,
        latitude: coord.latitude,
        longitude: coord.longitude,
        title: coord.location_name,
        threat_level: coord.threat_level,
      }));

      setPoints(newPoints);
    } else {
      setPoints([]);
    }
  }, [analysisData]);

  const fetchCrimeTypes = async () => {
    try {
      const response = await fetch('http://localhost:8000/crime-types');
      if (response.ok) {
        const data = await response.json();
        const crimeTypes = data.crime_types || [];

        // Get counts for each crime type (simplified)
        const crimeTypeOptions: CrimeTypeOption[] = crimeTypes.map((type: string) => ({
          name: type,
          count: 0 // Will be populated by states data
        }));

        setAvailableCrimeTypes(crimeTypeOptions);
      }
    } catch (err) {
      console.error('Error fetching crime types:', err);
    }
  };

  const fetchNigerianStatesData = async () => {
    try {
      setLoading(true);
      setError(null);

      // UPDATED: Add crime type parameter to URL
      const url = selectedCrimeTypeFilter
        ? `http://localhost:8000/nigerian-states-incidents?crime_type=${encodeURIComponent(selectedCrimeTypeFilter)}`
        : 'http://localhost:8000/nigerian-states-incidents';

      console.log('Fetching states data with URL:', url);

      const response = await fetch(url);

      if (response.ok) {
        const data = await response.json();
        if (data.status === 'success') {
          setStatesData(data.states_data || []);
          console.log('Nigerian states data loaded for crime type:', selectedCrimeTypeFilter || 'All', data.states_data);
        } else {
          setError(data.error || 'Failed to load states data');
        }
      } else {
        setError('Failed to fetch Nigerian states data');
      }
    } catch (err) {
      console.error('Error fetching Nigerian states data:', err);
      setError('Error loading map data');
    } finally {
      setLoading(false);
    }
  };

  // UPDATED: Remove dummy filtering since backend now handles it
  const getFilteredStatesData = () => {
    const data = statesData; // Data is already filtered by backend

    switch (viewMode) {
      case 'high_threat':
        return data.filter(state => state.threat_level === 'high');
      case 'recent':
        return data.slice(0, 10); // Top 10 by incidents (already sorted)
      default:
        return data;
    }
  };

  const getStateMarkerOptions = (state: NigerianStateData) => {
    const baseRadius = Math.max(8, Math.min(25, 8 + (state.incidents / 20)));

    switch (state.threat_level) {
      case 'high':
        return {
          radius: baseRadius,
          fillColor: '#dc2626',
          color: '#7f1d1d',
          weight: 2,
          fillOpacity: 0.8,
          className: 'high-threat-marker'
        };
      case 'medium':
        return {
          radius: baseRadius,
          fillColor: '#ea580c',
          color: '#9a3412',
          weight: 2,
          fillOpacity: 0.7,
          className: 'medium-threat-marker'
        };
      case 'low':
      default:
        return {
          radius: baseRadius,
          fillColor: '#16a34a',
          color: '#15803d',
          weight: 1,
          fillOpacity: 0.6,
          className: 'low-threat-marker'
        };
    }
  };

  const getDocumentPointOptions = (threatLevel: 'low' | 'medium' | 'high') => {
    switch (threatLevel) {
      case 'high':
        return { radius: 12, fillColor: '#ef4444', color: '#b91c1c', weight: 2, fillOpacity: 0.7 };
      case 'medium':
        return { radius: 9, fillColor: '#f59e0b', color: '#b45309', weight: 1, fillOpacity: 0.7 };
      case 'low':
      default:
        return { radius: 6, fillColor: '#10b981', color: '#047857', weight: 1, fillOpacity: 0.7 };
    }
  };

  const getTopStates = () => {
    return getFilteredStatesData().slice(0, 5).sort((a, b) => b.incidents - a.incidents);
  };

  const getThreatLevelStats = () => {
    const filtered = getFilteredStatesData();
    return {
      high: filtered.filter(s => s.threat_level === 'high').length,
      medium: filtered.filter(s => s.threat_level === 'medium').length,
      low: filtered.filter(s => s.threat_level === 'low').length,
      total: filtered.length
    };
  };

  const displayedStatesData = getFilteredStatesData();
  const threatStats = getThreatLevelStats();

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 flex items-center">
            <MapPin className="h-6 w-6 text-red-600 mr-2" />
            Multi-Crime Nigerian States Intelligence Map
          </h2>
          <p className="text-gray-600 mt-1">
            Interactive security intelligence visualization across all crime types and Nigerian states
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={fetchNigerianStatesData}
            className="flex items-center px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            disabled={loading}
          >
            <RefreshCw className={`h-4 w-4 mr-1 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Enhanced Map Controls */}
      <div className="bg-white p-6 rounded-lg shadow border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Filter className="h-5 w-5 text-blue-600 mr-2" />
          Map Controls & Filters
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          {/* Crime Type Filter - UPDATED */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Crime Type Focus</label>
            <select
              value={selectedCrimeTypeFilter}
              onChange={(e) => setSelectedCrimeTypeFilter(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            >
              <option value="">All Crime Types</option>
              {availableCrimeTypes.map(type => (
                <option key={type.name} value={type.name}>{type.name}</option>
              ))}
            </select>
          </div>

          {/* View Mode */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">View Mode</label>
            <select
              value={viewMode}
              onChange={(e) => setViewMode(e.target.value as any)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All States</option>
              <option value="high_threat">High Threat Only</option>
              <option value="recent">Top 10 Most Affected</option>
            </select>
          </div>

          {/* Layer Toggles */}
          <div className="space-y-2">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="showStates"
                checked={showStates}
                onChange={(e) => setShowStates(e.target.checked)}
                className="mr-2"
              />
              <label htmlFor="showStates" className="text-sm font-medium">
                State Incidents ({displayedStatesData.length})
              </label>
            </div>
            <div className="flex items-center">
              <input
                type="checkbox"
                id="showDocumentPoints"
                checked={showDocumentPoints}
                onChange={(e) => setShowDocumentPoints(e.target.checked)}
                className="mr-2"
              />
              <label htmlFor="showDocumentPoints" className="text-sm font-medium">
                Document Points ({points.length})
              </label>
            </div>
          </div>

          {/* Legend */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Threat Levels</label>
            <div className="space-y-1">
              <div className="flex items-center text-xs">
                <div className="w-3 h-3 bg-red-600 rounded-full mr-2"></div>
                <span>High Threat ({threatStats.high})</span>
              </div>
              <div className="flex items-center text-xs">
                <div className="w-3 h-3 bg-orange-600 rounded-full mr-2"></div>
                <span>Medium Threat ({threatStats.medium})</span>
              </div>
              <div className="flex items-center text-xs">
                <div className="w-3 h-3 bg-green-600 rounded-full mr-2"></div>
                <span>Low Threat ({threatStats.low})</span>
              </div>
            </div>
          </div>
        </div>

        {/* Active Filters Display */}
        {(selectedCrimeTypeFilter || viewMode !== 'all') && (
          <div className="flex flex-wrap gap-2 pt-3 border-t border-gray-200">
            {selectedCrimeTypeFilter && (
              <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                Crime Type: {selectedCrimeTypeFilter}
              </span>
            )}
            {viewMode !== 'all' && (
              <span className="px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded-full">
                View: {viewMode.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Enhanced Statistics Panel */}
      {displayedStatesData.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white p-4 rounded-lg shadow border">
            <div className="flex items-center">
              <Target className="h-5 w-5 text-blue-600 mr-2" />
              <div>
                <p className="text-sm text-gray-600">States Shown</p>
                <p className="text-xl font-bold text-blue-600">{displayedStatesData.length}</p>
                <p className="text-xs text-gray-500">
                  {selectedCrimeTypeFilter ? `for ${selectedCrimeTypeFilter}` : 'all crime types'}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow border">
            <div className="flex items-center">
              <TrendingUp className="h-5 w-5 text-red-600 mr-2" />
              <div>
                <p className="text-sm text-gray-600">Total Incidents</p>
                <p className="text-xl font-bold text-red-600">
                  {displayedStatesData.reduce((sum, state) => sum + state.incidents, 0).toLocaleString()}
                </p>
                <p className="text-xs text-gray-500">
                  {selectedCrimeTypeFilter || 'All types'}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow border">
            <div className="flex items-center">
              <AlertTriangle className="h-5 w-5 text-orange-600 mr-2" />
              <div>
                <p className="text-sm text-gray-600">High Threat States</p>
                <p className="text-xl font-bold text-orange-600">
                  {threatStats.high}
                </p>
                <p className="text-xs text-gray-500">Require immediate attention</p>
              </div>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow border">
            <div className="flex items-center">
              <Shield className="h-5 w-5 text-purple-600 mr-2" />
              <div>
                <p className="text-sm text-gray-600">Most Affected</p>
                <p className="text-lg font-bold text-purple-600">
                  {getTopStates()[0]?.name || 'N/A'}
                </p>
                <p className="text-xs text-gray-500">
                  {getTopStates()[0]?.incidents.toLocaleString() || 0} incidents
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center">
          <AlertTriangle className="h-5 w-5 text-red-600 mr-3" />
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {/* Interactive Map */}
      <div className="bg-gray-200 rounded-lg shadow border overflow-hidden relative h-[70vh]">
        <MapContainer center={nigeriaCenter} zoom={6} style={{ height: '100%', width: '100%' }} scrollWheelZoom={true}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {/* Nigerian States Markers */}
          {showStates && displayedStatesData.map(state => {
            const options = getStateMarkerOptions(state);
            return (
              <CircleMarker
                key={`state-${state.id}`}
                center={[state.latitude, state.longitude]}
                pathOptions={options}
                radius={options.radius}
              >
                <Popup>
                  <div className="min-w-[250px]">
                    <div className="font-bold text-lg text-gray-900 mb-2">{state.name}</div>
                    <div className="space-y-1 text-sm">
                      <div><span className="font-medium">Capital:</span> {state.capital}</div>
                      <div><span className="font-medium">Incidents:</span> <span className="text-red-600 font-bold">{state.incidents.toLocaleString()}</span></div>
                      <div><span className="font-medium">Document Mentions:</span> {state.mentions}</div>
                      <div>
                        <span className="font-medium">Threat Level:</span>
                        <span className={`ml-1 px-2 py-1 rounded text-xs font-bold ${
                          state.threat_level === 'high' ? 'bg-red-100 text-red-800' :
                          state.threat_level === 'medium' ? 'bg-orange-100 text-orange-800' :
                          'bg-green-100 text-green-800'
                        }`}>
                          {state.threat_level.toUpperCase()}
                        </span>
                      </div>
                      {selectedCrimeTypeFilter && (
                        <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded">
                          <span className="text-xs text-blue-800 font-medium">
                            Filtered by: {selectedCrimeTypeFilter}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}

          {/* Document-based Points */}
          {showDocumentPoints && points.map(point => {
            const options = getDocumentPointOptions(point.threat_level);
            return (
              <CircleMarker
                key={`doc-${point.id}`}
                center={[point.latitude, point.longitude]}
                pathOptions={options}
                radius={options.radius}
              >
                <Popup>
                  <div>
                    <div className="font-bold">{point.title}</div>
                    <div>Document Threat Level: {point.threat_level.toUpperCase()}</div>
                    <div className="text-xs text-gray-600 mt-1">From document analysis</div>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}

          <MapBoundsUpdater points={points} statesData={displayedStatesData} />
        </MapContainer>

        {loading && (
          <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-1000">
            <div className="bg-white p-4 rounded-lg flex items-center">
              <RefreshCw className="h-5 w-5 animate-spin mr-2" />
              <span>Loading crime-specific intelligence data...</span>
            </div>
          </div>
        )}
      </div>

      {/* Enhanced Top States Table */}
      {displayedStatesData.length > 0 && (
        <div className="bg-white rounded-lg shadow border p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Database className="h-5 w-5 text-blue-600 mr-2" />
            Most Affected States
            {selectedCrimeTypeFilter && (
              <span className="ml-2 text-sm text-purple-600">- {selectedCrimeTypeFilter}</span>
            )}
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rank</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">State</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Capital</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Incidents</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Threat Level</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Mentions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {getTopStates().map((state, index) => (
                  <tr key={state.id} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">#{index + 1}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{state.name}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{state.capital}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-red-600">{state.incidents.toLocaleString()}</td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        state.threat_level === 'high' ? 'bg-red-100 text-red-800' :
                        state.threat_level === 'medium' ? 'bg-orange-100 text-orange-800' :
                        'bg-green-100 text-green-800'
                      }`}>
                        {state.threat_level}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{state.mentions}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {selectedCrimeTypeFilter && (
            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>Note:</strong> Data filtered for {selectedCrimeTypeFilter}.
                Switch to "All Crime Types" to see comprehensive state rankings.
              </p>
            </div>
          )}
        </div>
      )}

      {/* No Data State */}
      {!loading && displayedStatesData.length === 0 && (
        <div className="text-center bg-gray-50 border border-gray-200 rounded-lg p-8">
          <MapPin className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No Geographic Data Found
          </h3>
          <p className="text-gray-600">
            {selectedCrimeTypeFilter
              ? `No data found for ${selectedCrimeTypeFilter}. Try selecting a different crime type or "All Crime Types".`
              : 'Upload intelligence documents with geographic mentions to see state-level analysis.'
            }
          </p>
          <div className="mt-4 flex justify-center gap-2">
            <button
              onClick={fetchNigerianStatesData}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Refresh Data
            </button>
            {selectedCrimeTypeFilter && (
              <button
                onClick={() => setSelectedCrimeTypeFilter('')}
                className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
              >
                Show All Crime Types
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default GeospatialMap;