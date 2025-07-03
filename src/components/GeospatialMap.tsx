// src/components/GeospatialMap.tsx - FINAL CORRECTED VERSION

import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L, { LatLngExpression } from 'leaflet'; // <<< FIX: Import the main Leaflet object 'L'

import { IntelligenceDocument } from '../App';
import { MapPin, AlertTriangle } from 'lucide-react';

interface GeospatialPoint {
  id: string;
  latitude: number;
  longitude: number;
  title: string;
  threat_level: 'low' | 'medium' | 'high';
}

interface GeospatialMapProps {
  analysisData?: IntelligenceDocument | null;
}

// Helper component to automatically adjust map bounds
const MapBoundsUpdater: React.FC<{ points: GeospatialPoint[] }> = ({ points }) => {
    const map = useMap();
    useEffect(() => {
      if (points.length > 0) {
        const boundsArray = points.map(p => [p.latitude, p.longitude] as LatLngExpression);
        // <<< FIX: Create a LatLngBounds object using L.latLngBounds() >>>
        const bounds = L.latLngBounds(boundsArray);
        map.fitBounds(bounds, { padding: [50, 50] });
      }
    }, [points, map]);
    return null;
};


const GeospatialMap: React.FC<GeospatialMapProps> = ({ analysisData }) => {
  const [points, setPoints] = useState<GeospatialPoint[]>([]);
  const nigeriaCenter: LatLngExpression = [9.0820, 8.6753];

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

  const getPathOptions = (threatLevel: 'low' | 'medium' | 'high') => {
    switch (threatLevel) {
      case 'high':
        return { radius: 15, fillColor: '#ef4444', color: '#b91c1c', weight: 2, fillOpacity: 0.7 };
      case 'medium':
        return { radius: 10, fillColor: '#f59e0b', color: '#b45309', weight: 1, fillOpacity: 0.7 };
      case 'low':
      default:
        return { radius: 7, fillColor: '#10b981', color: '#047857', weight: 1, fillOpacity: 0.7 };
    }
  };

  return (
    <div className="p-6 space-y-6">
       <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 flex items-center">
            <MapPin className="h-6 w-6 text-red-600 mr-2" />
            Geospatial Intelligence Map
          </h2>
          <p className="text-gray-600 mt-1">
            {analysisData ? `Threat hotspots from: ${analysisData.metadata.filename}` : 'Interactive threat map'}
          </p>
        </div>
      </div>

      {!analysisData && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 flex items-center">
          <AlertTriangle className="h-5 w-5 text-yellow-600 mr-3" />
          <p className="text-yellow-700">Upload a document to see threat hotspots on the map.</p>
        </div>
      )}

      <div className="bg-gray-200 rounded-lg shadow border overflow-hidden relative h-[70vh]">
        <MapContainer center={nigeriaCenter} zoom={6} style={{ height: '100%', width: '100%' }} scrollWheelZoom={true}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {points.map(point => {
            const options = getPathOptions(point.threat_level);
            return (
              <CircleMarker
                key={point.id}
                center={[point.latitude, point.longitude]}
                pathOptions={options}
                // <<< FIX: Explicitly pass the 'radius' prop to satisfy the type definition >>>
                radius={options.radius}
              >
                <Popup>
                  <div className="font-bold">{point.title}</div>
                  <div>Threat Level: {point.threat_level.toUpperCase()}</div>
                </Popup>
              </CircleMarker>
            );
          })}

          <MapBoundsUpdater points={points} />
        </MapContainer>
      </div>
    </div>
  );
};

export default GeospatialMap;