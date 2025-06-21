// src/components/GeospatialMap.tsx - Interactive Geospatial Intelligence Map

import React, { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import { MapPin, Layers, Filter, Download, Maximize2, Search, AlertTriangle } from 'lucide-react';

// Set Mapbox access token
mapboxgl.accessToken = 'pk.eyJ1IjoiYWJkdWxkYXR0aWpvMSIsImEiOiJjbWM2bW83Y3IwbmN4MmtzYWw0cHppbXF4In0.0pSWoI2VOyPyORqCROJc9g';

interface GeospatialPoint {
  id: string;
  latitude: number;
  longitude: number;
  title: string;
  description: string;
  threat_level: 'low' | 'medium' | 'high';
  type: 'incident' | 'surveillance' | 'intelligence' | 'facility' | 'location';
  timestamp: string;
  metadata?: Record<string, any>;
}

interface MapFilter {
  threat_levels: string[];
  point_types: string[];
  date_range: {
    start: string;
    end: string;
  };
}

interface Document {
  id: string;
  content: string;
  metadata: {
    filename: string;
    file_type: string;
    uploaded_at: string;
    file_size: number;
  };
  analysis: {
    geographic_intelligence: {
      states: string[];
      cities: string[];
      countries: string[];
      coordinates: Array<{
        latitude: number;
        longitude: number;
        location_name: string;
        confidence: number;
      }>;
      total_locations: number;
      other_locations: string[];
    };
    sentiment_analysis: {
      threat_level: string;
    };
    entities: {
      locations: string[];
    };
  };
}

interface GeospatialMapProps {
  analysisData?: Document | null;
}

const GeospatialMap: React.FC<GeospatialMapProps> = ({ analysisData }) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [selectedPoint, setSelectedPoint] = useState<GeospatialPoint | null>(null);
  const [points, setPoints] = useState<GeospatialPoint[]>([]);
  const [filters, setFilters] = useState<MapFilter>({
    threat_levels: ['low', 'medium', 'high'],
    point_types: ['incident', 'surveillance', 'intelligence', 'facility', 'location'],
    date_range: {
      start: '',
      end: ''
    }
  });
  const [showFilters, setShowFilters] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    if (!mapContainer.current) return;

    // Initialize map
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/satellite-v9',
      center: [7.4, 9.0], // Nigeria center
      zoom: 6,
      pitch: 0,
      bearing: 0
    });

    map.current.on('load', () => {
      setIsLoaded(true);
      setupMapLayers();
    });

    // Add navigation controls
    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

    // Add fullscreen control
    map.current.addControl(new mapboxgl.FullscreenControl(), 'top-right');

    return () => {
      if (map.current) {
        map.current.remove();
      }
    };
  }, []);

  useEffect(() => {
    // Load data from analysis when it changes
    if (analysisData) {
      loadAnalysisData();
    } else {
      loadMockData();
    }
  }, [analysisData]);

  const loadAnalysisData = () => {
    if (!analysisData) return;

    const analysisPoints: GeospatialPoint[] = [];
    const geoIntel = analysisData.analysis.geographic_intelligence;
    const threatLevel = analysisData.analysis.sentiment_analysis.threat_level.toLowerCase() as 'low' | 'medium' | 'high';

    // Add coordinates from analysis
    geoIntel.coordinates.forEach((coord, index) => {
      analysisPoints.push({
        id: `coord_${index}`,
        latitude: coord.latitude,
        longitude: coord.longitude,
        title: coord.location_name || 'Identified Location',
        description: `Location extracted from document analysis with ${(coord.confidence * 100).toFixed(1)}% confidence`,
        threat_level: threatLevel,
        type: 'location',
        timestamp: analysisData.metadata.uploaded_at,
        metadata: {
          confidence: coord.confidence,
          source: 'document_analysis',
          filename: analysisData.metadata.filename
        }
      });
    });

    // Create points for major Nigerian cities mentioned in states
    const nigerianCities = {
      'lagos': { lat: 6.5244, lng: 3.3792 },
      'abuja': { lat: 9.0765, lng: 7.3986 },
      'kano': { lat: 12.0022, lng: 8.5920 },
      'ibadan': { lat: 7.3775, lng: 3.9470 },
      'port harcourt': { lat: 4.8156, lng: 7.0498 },
      'kaduna': { lat: 10.5105, lng: 7.4165 },
      'benin city': { lat: 6.3350, lng: 5.6037 },
      'maiduguri': { lat: 11.8311, lng: 13.1510 },
      'jos': { lat: 9.8965, lng: 8.8583 },
      'ilorin': { lat: 8.5000, lng: 4.5500 }
    };

    // Add points for mentioned states and cities
    const allLocations = [
      ...geoIntel.states,
      ...geoIntel.cities,
      ...geoIntel.other_locations
    ];

    allLocations.forEach((location, index) => {
      const locationKey = location.toLowerCase().trim();
      const cityData = nigerianCities[locationKey as keyof typeof nigerianCities];

      if (cityData) {
        analysisPoints.push({
          id: `location_${index}`,
          latitude: cityData.lat,
          longitude: cityData.lng,
          title: `${location} (Mentioned)`,
          description: `Location "${location}" mentioned in document: ${analysisData.metadata.filename}`,
          threat_level: threatLevel,
          type: 'intelligence',
          timestamp: analysisData.metadata.uploaded_at,
          metadata: {
            source: 'location_mention',
            original_text: location,
            filename: analysisData.metadata.filename
          }
        });
      }
    });

    // If no specific coordinates but locations mentioned, focus on Nigeria
    if (analysisPoints.length === 0 && allLocations.length > 0) {
      // Add a general Nigeria point
      analysisPoints.push({
        id: 'nigeria_general',
        latitude: 9.0820,
        longitude: 8.6753,
        title: 'Nigeria (General Reference)',
        description: `Document mentions Nigerian locations but no specific coordinates found. Locations: ${allLocations.slice(0, 3).join(', ')}${allLocations.length > 3 ? '...' : ''}`,
        threat_level: threatLevel,
        type: 'intelligence',
        timestamp: analysisData.metadata.uploaded_at,
        metadata: {
          source: 'general_reference',
          mentioned_locations: allLocations,
          filename: analysisData.metadata.filename
        }
      });
    }

    setPoints(analysisPoints);

    // Adjust map view to show all points
    if (analysisPoints.length > 0 && map.current) {
      const bounds = new mapboxgl.LngLatBounds();
      analysisPoints.forEach(point => {
        bounds.extend([point.longitude, point.latitude]);
      });

      setTimeout(() => {
        if (map.current) {
          map.current.fitBounds(bounds, { padding: 50, maxZoom: 12 });
        }
      }, 1000);
    }
  };

  const loadMockData = () => {
    // Fallback mock data when no analysis data is available
    const mockPoints: GeospatialPoint[] = [
      {
        id: '1',
        latitude: 6.5244,
        longitude: 3.3792,
        title: 'Lagos Security Incident',
        description: 'Suspicious activity reported in Victoria Island area',
        threat_level: 'high',
        type: 'incident',
        timestamp: '2025-06-20T10:30:00Z',
        metadata: { severity: 8, casualties: 0 }
      },
      {
        id: '2',
        latitude: 9.0765,
        longitude: 7.3986,
        title: 'Abuja Intelligence Facility',
        description: 'Central intelligence coordination center',
        threat_level: 'low',
        type: 'facility',
        timestamp: '2025-06-20T08:00:00Z',
        metadata: { classification: 'confidential' }
      },
      {
        id: '3',
        latitude: 7.3775,
        longitude: 3.9470,
        title: 'Ibadan Surveillance Point',
        description: 'Active monitoring station for regional intelligence',
        threat_level: 'medium',
        type: 'surveillance',
        timestamp: '2025-06-20T14:15:00Z',
        metadata: { coverage_radius: 25 }
      }
    ];

    setPoints(mockPoints);
  };

  const setupMapLayers = () => {
    if (!map.current || !isLoaded) return;

    // Add data source
    map.current.addSource('intelligence-points', {
      type: 'geojson',
      data: {
        type: 'FeatureCollection',
        features: []
      }
    });

    // Add circle layer for points
    map.current.addLayer({
      id: 'intelligence-circles',
      type: 'circle',
      source: 'intelligence-points',
      paint: {
        'circle-radius': [
          'case',
          ['==', ['get', 'threat_level'], 'high'], 12,
          ['==', ['get', 'threat_level'], 'medium'], 8,
          6
        ],
        'circle-color': [
          'case',
          ['==', ['get', 'threat_level'], 'high'], '#ef4444',
          ['==', ['get', 'threat_level'], 'medium'], '#f59e0b',
          '#10b981'
        ],
        'circle-opacity': 0.8,
        'circle-stroke-width': 2,
        'circle-stroke-color': '#ffffff'
      }
    });

    // Add click event
    map.current.on('click', 'intelligence-circles', (e) => {
      if (e.features && e.features[0]) {
        const feature = e.features[0];
        const point = points.find(p => p.id === feature.properties?.id);
        if (point) {
          setSelectedPoint(point);

          // Create popup
          new mapboxgl.Popup()
            .setLngLat([point.longitude, point.latitude])
            .setHTML(`
              <div class="p-3 max-w-sm">
                <h3 class="font-semibold text-gray-900 mb-2">${point.title}</h3>
                <p class="text-sm text-gray-600 mb-2">${point.description}</p>
                <div class="flex items-center justify-between">
                  <span class="px-2 py-1 rounded text-xs font-medium ${
                    point.threat_level === 'high' ? 'bg-red-100 text-red-800' :
                    point.threat_level === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-green-100 text-green-800'
                  }">
                    ${point.threat_level.toUpperCase()}
                  </span>
                  <span class="text-xs text-gray-500">${point.type}</span>
                </div>
                ${point.metadata?.filename ? `<p class="text-xs text-gray-400 mt-1">Source: ${point.metadata.filename}</p>` : ''}
              </div>
            `)
            .addTo(map.current!);
        }
      }
    });

    // Change cursor on hover
    map.current.on('mouseenter', 'intelligence-circles', () => {
      if (map.current) {
        map.current.getCanvas().style.cursor = 'pointer';
      }
    });

    map.current.on('mouseleave', 'intelligence-circles', () => {
      if (map.current) {
        map.current.getCanvas().style.cursor = '';
      }
    });
  };
  useEffect(() => {
    if (!map.current || !isLoaded) return;

    const filteredPoints = points.filter(point => {
      const matchesThreatLevel = filters.threat_levels.includes(point.threat_level);
      const matchesType = filters.point_types.includes(point.type);

      let matchesDateRange = true;
      if (filters.date_range.start && filters.date_range.end) {
        const pointDate = new Date(point.timestamp);
        const startDate = new Date(filters.date_range.start);
        const endDate = new Date(filters.date_range.end);
        matchesDateRange = pointDate >= startDate && pointDate <= endDate;
      }

      return matchesThreatLevel && matchesType && matchesDateRange;
    });

    const geojsonData = {
      type: 'FeatureCollection' as const,
      features: filteredPoints.map(point => ({
        type: 'Feature' as const,
        properties: {
          id: point.id,
          title: point.title,
          description: point.description,
          threat_level: point.threat_level,
          type: point.type,
          timestamp: point.timestamp
        },
        geometry: {
          type: 'Point' as const,
          coordinates: [point.longitude, point.latitude]
        }
      }))
    };

    const source = map.current.getSource('intelligence-points') as mapboxgl.GeoJSONSource;
    if (source) {
      source.setData(geojsonData);
    }
  }, [points, filters, isLoaded]);

  const toggleFilter = (filterType: keyof MapFilter, value: string) => {
    setFilters(prev => ({
      ...prev,
      [filterType]: Array.isArray(prev[filterType])
        ? (prev[filterType] as string[]).includes(value)
          ? (prev[filterType] as string[]).filter(item => item !== value)
          : [...(prev[filterType] as string[]), value]
        : prev[filterType]
    }));
  };

  const exportMapData = () => {
    const dataStr = JSON.stringify(points, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'intelligence_map_data.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  const getThreatLevelColor = (level: string) => {
    switch (level) {
      case 'high': return 'text-red-600 bg-red-50 border-red-200';
      case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      default: return 'text-green-600 bg-green-50 border-green-200';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 flex items-center">
            <MapPin className="h-6 w-6 text-red-600 mr-2" />
            Geospatial Intelligence Map
          </h2>
          <p className="text-gray-600 mt-1">
            {analysisData
              ? `Geographic intelligence from: ${analysisData.metadata.filename}`
              : 'Interactive map with intelligence points and threat analysis'
            }
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
              showFilters ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            <Filter className="h-4 w-4" />
            <span>Filters</span>
          </button>

          <button
            onClick={exportMapData}
            className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
          >
            <Download className="h-4 w-4" />
            <span>Export</span>
          </button>
        </div>
      </div>

      {/* Analysis Data Summary */}
      {analysisData && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-blue-900 mb-2">Document Analysis Summary</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-blue-700 font-medium">Total Locations:</span>
              <span className="ml-2 text-blue-900">{analysisData.analysis.geographic_intelligence.total_locations}</span>
            </div>
            <div>
              <span className="text-blue-700 font-medium">States:</span>
              <span className="ml-2 text-blue-900">{analysisData.analysis.geographic_intelligence.states.length}</span>
            </div>
            <div>
              <span className="text-blue-700 font-medium">Coordinates:</span>
              <span className="ml-2 text-blue-900">{analysisData.analysis.geographic_intelligence.coordinates.length}</span>
            </div>
            <div>
              <span className="text-blue-700 font-medium">Threat Level:</span>
              <span className="ml-2 text-blue-900">{analysisData.analysis.sentiment_analysis.threat_level}</span>
            </div>
          </div>
        </div>
      )}

      {/* No Data Warning */}
      {!analysisData && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 text-yellow-600 mr-2" />
            <h3 className="text-yellow-800 font-medium">No Document Analysis Data</h3>
          </div>
          <p className="text-yellow-700 mt-1">
            Upload and analyze a document first to see real geographic intelligence data on the map.
            Currently showing sample data.
          </p>
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg shadow border">
          <div className="text-2xl font-bold text-blue-600">{points.length}</div>
          <div className="text-sm text-gray-500">Total Points</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow border">
          <div className="text-2xl font-bold text-red-600">
            {points.filter(p => p.threat_level === 'high').length}
          </div>
          <div className="text-sm text-gray-500">High Threat</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow border">
          <div className="text-2xl font-bold text-yellow-600">
            {points.filter(p => p.threat_level === 'medium').length}
          </div>
          <div className="text-sm text-gray-500">Medium Threat</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow border">
          <div className="text-2xl font-bold text-green-600">
            {points.filter(p => p.threat_level === 'low').length}
          </div>
          <div className="text-sm text-gray-500">Low Threat</div>
        </div>
      </div>

      {/* Filters Panel */}
      {showFilters && (
        <div className="bg-white p-6 rounded-lg shadow border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Map Filters</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Threat Level Filter */}
            <div>
              <h4 className="font-medium text-gray-900 mb-2">Threat Levels</h4>
              <div className="space-y-2">
                {['high', 'medium', 'low'].map((level) => (
                  <label key={level} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={filters.threat_levels.includes(level)}
                      onChange={() => toggleFilter('threat_levels', level)}
                      className="mr-2 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className={`px-2 py-1 rounded text-xs font-medium ${getThreatLevelColor(level)}`}>
                      {level.toUpperCase()}
                    </span>
                  </label>
                ))}
              </div>
            </div>

            {/* Point Type Filter */}
            <div>
              <h4 className="font-medium text-gray-900 mb-2">Point Types</h4>
              <div className="space-y-2">
                {['incident', 'surveillance', 'intelligence', 'facility', 'location'].map((type) => (
                  <label key={type} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={filters.point_types.includes(type)}
                      onChange={() => toggleFilter('point_types', type)}
                      className="mr-2 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="capitalize">{type}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Date Range Filter */}
            <div>
              <h4 className="font-medium text-gray-900 mb-2">Date Range</h4>
              <div className="space-y-2">
                <input
                  type="date"
                  value={filters.date_range.start}
                  onChange={(e) => setFilters(prev => ({
                    ...prev,
                    date_range: { ...prev.date_range, start: e.target.value }
                  }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  placeholder="Start Date"
                />
                <input
                  type="date"
                  value={filters.date_range.end}
                  onChange={(e) => setFilters(prev => ({
                    ...prev,
                    date_range: { ...prev.date_range, end: e.target.value }
                  }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  placeholder="End Date"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Map Container */}
      <div className="bg-white rounded-lg shadow border overflow-hidden">
        <div
          ref={mapContainer}
          className="h-96 w-full"
          style={{ minHeight: '600px' }}
        />

        {!isLoaded && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-50">
            <div className="text-center">
              <MapPin className="h-8 w-8 text-blue-600 mx-auto mb-2 animate-pulse" />
              <p className="text-gray-600">Loading map...</p>
            </div>
          </div>
        )}
      </div>

      {/* Selected Point Details */}
      {selectedPoint && (
        <div className="bg-white p-6 rounded-lg shadow border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <AlertTriangle className="h-5 w-5 text-blue-600 mr-2" />
            Selected Point Details
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-medium text-gray-900">{selectedPoint.title}</h4>
              <p className="text-gray-600 mt-1">{selectedPoint.description}</p>
              <div className="mt-3 flex items-center space-x-4">
                <span className={`px-2 py-1 rounded text-xs font-medium ${getThreatLevelColor(selectedPoint.threat_level)}`}>
                  {selectedPoint.threat_level.toUpperCase()}
                </span>
                <span className="text-sm text-gray-500 capitalize">{selectedPoint.type}</span>
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-500">
                <p><strong>Coordinates:</strong> {selectedPoint.latitude.toFixed(4)}, {selectedPoint.longitude.toFixed(4)}</p>
                <p><strong>Timestamp:</strong> {new Date(selectedPoint.timestamp).toLocaleString()}</p>
                {selectedPoint.metadata && (
                  <div className="mt-2">
                    <strong>Metadata:</strong>
                    <ul className="ml-4 mt-1">
                      {Object.entries(selectedPoint.metadata).map(([key, value]) => (
                        <li key={key} className="text-xs">
                          {key}: {String(value)}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GeospatialMap;