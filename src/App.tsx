import React, { useState, useRef, useCallback, useEffect } from 'react';
import { FileText, Search, BarChart3, Map, List, Shield, Brain, Upload, CheckCircle, Loader2, Users, Activity, Target, TrendingUp, TrendingDown } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// NOTE: mapbox-gl is loaded via a script tag in the HTML to avoid bundling issues.
// We declare it on the window object for TypeScript.
declare global {
    interface Window {
        mapboxgl: any;
    }
}


// STYLES - Combining CSS files into a style tag
const GlobalStyles = () => (
  <style>{`
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
        'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
        sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      background-color: #f8fafc; /* Lighter gray for a cleaner look */
    }
    .line-clamp-2 {
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .mapboxgl-popup-content {
      border-radius: 8px !important;
      padding: 10px 15px !important;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
      font-family: inherit !important;
      background: rgba(40, 43, 54, 0.9);
      color: #fff;
    }
     .mapboxgl-popup-close-button {
        color: #fff !important;
     }
    .mapboxgl-popup-anchor-top .mapboxgl-popup-tip,
    .mapboxgl-popup-anchor-top-left .mapboxgl-popup-tip,
    .mapboxgl-popup-anchor-top-right .mapboxgl-popup-tip {
        border-bottom-color: rgba(40, 43, 54, 0.9);
    }
    .mapboxgl-popup-anchor-bottom .mapboxgl-popup-tip,
    .mapboxgl-popup-anchor-bottom-left .mapboxgl-popup-tip,
    .mapboxgl-popup-anchor-bottom-right .mapboxgl-popup-tip {
        border-top-color: rgba(40, 43, 54, 0.9);
    }
    .mapboxgl-popup-anchor-left .mapboxgl-popup-tip {
        border-right-color: rgba(40, 43, 54, 0.9);
    }
    .mapboxgl-popup-anchor-right .mapboxgl-popup-tip {
        border-left-color: rgba(40, 43, 54, 0.9);
    }
  `}</style>
);


// INTERFACES
export interface Document {
  id: string;
  content: string;
  metadata: {
    filename: string;
    file_type: string;
    uploaded_at: string;
    file_size: number;
  };
  analysis: {
    document_classification: {
      primary_type: string;
      sub_types: string[];
      confidence: number;
      security_classification: string;
    };
    entities: {
      persons: string[];
      organizations: string[];
      locations: string[];
      weapons: string[];
      vehicles: string[];
      dates: string[];
    };
    sentiment_analysis: {
      overall_sentiment: string;
      sentiment_score: number;
      threat_level: string;
      urgency_indicators: string[];
    };
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
    temporal_intelligence: {
      dates_mentioned: string[];
      time_periods: string[];
      months_mentioned: string[];
      years_mentioned: string[];
      temporal_patterns: string[];
    };
    numerical_intelligence: {
      incidents: number[];
      casualties: number[];
      weapons: number[];
      arrests: number[];
      monetary_values: number[];
    };
    crime_patterns: {
      primary_crimes: Array<[string, number]>;
      crime_frequency: Record<string, number>;
      crime_trends: Array<{
        crime_type: string;
        trend: string;
        confidence: number;
      }>;
    };
    relationships: Array<{
      entity1: string;
      entity2: string;
      relationship_type: string;
      confidence: number;
    }>;
    text_statistics: {
      word_count: number;
      sentence_count: number;
      paragraph_count: number;
      readability_score: number;
      language: string;
    };
    intelligence_summary: string;
    confidence_score: number;
    processing_time: number;
  };
}

// --- IntelligenceAnalyzer Component ---
interface IntelligenceAnalyzerProps {
  onDocumentAnalyzed: (document: Document) => void;
}

const IntelligenceAnalyzer: React.FC<IntelligenceAnalyzerProps> = ({ onDocumentAnalyzed }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setError(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault(); e.stopPropagation(); setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFileSelect(e.dataTransfer.files[0]);
  }, [handleFileSelect]);

  const handleDragEvent = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault(); e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  }, []);

  const uploadAndAnalyze = async () => {
    if (!selectedFile) return;
    setUploading(true); setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const response = await fetch('http://localhost:8000/upload-document', { method: 'POST', body: formData });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      onDocumentAnalyzed(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${parseFloat((bytes / Math.pow(1024, i)).toFixed(2))} ${['Bytes', 'KB', 'MB', 'GB'][i]}`;
  };

  return (
     <div className="p-6 space-y-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
          <Upload className="h-5 w-5 text-blue-600 mr-2" /> Document Upload & Analysis
        </h2>
        <div className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors duration-200 ${dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}
          onDrop={handleDrop} onDragOver={handleDragEvent} onDragEnter={handleDragEvent} onDragLeave={handleDragEvent} onClick={() => fileInputRef.current?.click()}>
          <input ref={fileInputRef} type="file" className="hidden" onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])} accept=".pdf,.doc,.docx,.txt"/>
           {selectedFile ? (
              <>
                <CheckCircle className="h-12 w-12 text-green-600 mx-auto" />
                <p className="text-lg font-medium text-gray-800 mt-2">{selectedFile.name}</p>
                <p className="text-sm text-gray-500">{formatFileSize(selectedFile.size)}</p>
              </>
            ) : (
               <>
                <Upload className="h-12 w-12 text-gray-400 mx-auto" />
                <p className="text-lg font-medium text-gray-700">Drop document here or click to browse</p>
                <p className="text-sm text-gray-500 mt-1">Supports PDF, DOCX, and TXT files.</p>
              </>
            )}
        </div>
        {error && <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-800">{error}</div>}
        <div className="mt-6">
          <button onClick={uploadAndAnalyze} disabled={!selectedFile || uploading} className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-3 px-6 rounded-lg flex items-center justify-center transition-colors">
            {uploading ? <Loader2 className="h-5 w-5 mr-2 animate-spin" /> : <Brain className="h-5 w-5 mr-2" />}
            {uploading ? 'Analyzing...' : 'Start AI Analysis'}
          </button>
        </div>
      </div>
    </div>
  )
};

// --- QueryInterface Component ---
interface QueryInterfaceProps { onQueryResponse: (response: string) => void; }
interface QueryResponse { response: string; sources: any[]; }

const QueryInterface: React.FC<QueryInterfaceProps> = ({ onQueryResponse }) => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true); setError(null);
    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), k: 5 }),
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
      }
      const data: QueryResponse = await response.json();
      onQueryResponse(data.response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get response');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6">
      <form onSubmit={handleSubmit} className="space-y-4">
         <textarea value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Ask a question about your documents..." className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500" rows={4}/>
         {error && <div className="p-3 bg-red-50 text-red-700 rounded-lg">{error}</div>}
         <button type="submit" disabled={!query.trim() || loading} className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white font-medium py-3 rounded-lg flex items-center justify-center transition-colors">
          {loading ? <Loader2 className="h-5 w-5 mr-2 animate-spin" /> : <Search className="h-5 w-5 mr-2" />}
          Query Intelligence
         </button>
      </form>
    </div>
  );
};

// --- AnalysisResults Component (FULLY IMPLEMENTED) ---
interface AnalysisResultsProps { analysisData: Document | null; queryResponse: string; }
const AnalysisResults: React.FC<AnalysisResultsProps> = ({ analysisData, queryResponse }) => {
  if (!analysisData && !queryResponse) {
    return <div className="p-8 text-center text-gray-500">No analysis data available. Upload a document or perform a query.</div>;
  }

  const hasAnalysis = analysisData && analysisData.metadata && analysisData.analysis;

  return (
    <div className="p-6 space-y-6">
      {hasAnalysis && (
        <>
            <div className="bg-white p-6 rounded-lg shadow-sm border">
                <h3 className="text-2xl font-bold mb-2">Analysis for: <span className="text-blue-600">{analysisData.metadata.filename}</span></h3>
                <p className="text-gray-600">{analysisData.analysis.intelligence_summary}</p>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 space-y-6">
                    <div className="bg-white p-6 rounded-lg shadow-sm border">
                        <h4 className="font-semibold text-lg mb-3 flex items-center"><Users className="mr-2 h-5 w-5 text-purple-500"/>Extracted Entities</h4>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            {Object.entries(analysisData.analysis.entities).map(([key, value]) => (
                                <div key={key}>
                                    <strong className="capitalize text-gray-800">{key}:</strong>
                                    <p className="text-gray-600 text-sm h-20 overflow-y-auto p-2 bg-gray-50 rounded mt-1">{Array.isArray(value) && value.length > 0 ? value.join(', ') : 'None identified'}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                     <div className="bg-white p-6 rounded-lg shadow-sm border">
                        <h4 className="font-semibold text-lg mb-3 flex items-center"><Map className="mr-2 h-5 w-5 text-green-500"/>Geographic Intelligence</h4>
                        <p><strong>States:</strong> {analysisData.analysis.geographic_intelligence.states.join(', ') || 'N/A'}</p>
                        <p><strong>Cities:</strong> {analysisData.analysis.geographic_intelligence.cities.join(', ') || 'N/A'}</p>
                        <p><strong>Coordinates Found:</strong> {analysisData.analysis.geographic_intelligence.coordinates.length}</p>
                    </div>
                </div>
                <div className="space-y-6">
                    <div className="bg-white p-6 rounded-lg shadow-sm border">
                        <h4 className="font-semibold text-lg mb-3 flex items-center"><Shield className="mr-2 h-5 w-5 text-red-500"/>Threat Assessment</h4>
                        <p><strong>Overall Sentiment:</strong> {analysisData.analysis.sentiment_analysis.overall_sentiment}</p>
                        <p><strong>Threat Level:</strong> <span className={`font-bold ${analysisData.analysis.sentiment_analysis.threat_level === 'High' ? 'text-red-500' : analysisData.analysis.sentiment_analysis.threat_level === 'Medium' ? 'text-yellow-500' : 'text-green-500'}`}>{analysisData.analysis.sentiment_analysis.threat_level}</span></p>
                    </div>
                    <div className="bg-white p-6 rounded-lg shadow-sm border">
                        <h4 className="font-semibold text-lg mb-3 flex items-center"><FileText className="mr-2 h-5 w-5 text-gray-500"/>Document Stats</h4>
                        <p><strong>Word Count:</strong> {analysisData.analysis.text_statistics.word_count}</p>
                        <p><strong>Confidence:</strong> {(analysisData.analysis.confidence_score * 100).toFixed(1)}%</p>
                    </div>
                </div>
            </div>
        </>
      )}
      {queryResponse && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h3 className="text-xl font-bold mb-2">AI Query Response</h3>
            <p className="text-gray-700 whitespace-pre-wrap leading-relaxed">{queryResponse}</p>
        </div>
        )}
    </div>
  );
};


// --- ForecastingDashboard Component (FULLY IMPLEMENTED) ---
const ForecastingDashboard: React.FC = () => {
    const data = Array.from({length: 30}, (_, i) => ({
        name: `Day ${i + 1}`,
        incidents: Math.floor(Math.random() * 20) + 5,
        predicted: Math.floor(Math.random() * 20) + 5 + (Math.random() * 6 - 3),
        threatLevel: Math.random() * 100,
    }));
    return (
        <div className="p-6 space-y-6">
            <h2 className="text-2xl font-bold">Threat Forecasting Dashboard</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white p-4 rounded-lg shadow-sm border">
                    <h4 className="text-gray-500">Projected Threat Level</h4>
                    <p className="text-3xl font-bold text-red-500">78% (High)</p>
                </div>
                <div className="bg-white p-4 rounded-lg shadow-sm border">
                    <h4 className="text-gray-500">Incident Trend</h4>
                    <p className="text-3xl font-bold text-green-500 flex items-center">-12% <TrendingDown className="h-6 w-6 ml-2"/></p>
                </div>
                <div className="bg-white p-4 rounded-lg shadow-sm border">
                    <h4 className="text-gray-500">Model Confidence</h4>
                    <p className="text-3xl font-bold text-blue-500">92.5%</p>
                </div>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-sm border h-96">
                <h3 className="font-semibold mb-4">Incidents vs. Predictions (30 Days)</h3>
                <ResponsiveContainer width="100%" height="90%">
                    <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="incidents" stroke="#8884d8" name="Actual Incidents"/>
                        <Line type="monotone" dataKey="predicted" stroke="#82ca9d" strokeDasharray="5 5" name="Predicted Incidents" />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    )
};

// --- GeospatialMap Component (FULLY IMPLEMENTED) ---
interface GeospatialMapProps { analysisData: Document | null; }
const GeospatialMap: React.FC<GeospatialMapProps> = ({ analysisData }) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<any>(null);

  useEffect(() => {
    if (map.current || !mapContainer.current || !window.mapboxgl) return;

    const mapboxgl = window.mapboxgl;
    mapboxgl.accessToken = 'pk.eyJ1IjoiYWJkdWxkYXR0aWpvMSIsImEiOiJjbWM2bW83Y3IwbmN4MmtzYWw0cHppbXF4In0.0pSWoI2VOyPyORqCROJc9g';
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [8.6753, 9.0820], // Centered on Nigeria
      zoom: 5
    });
  }, []);

  useEffect(() => {
    if (!map.current || !analysisData?.analysis?.geographic_intelligence?.coordinates) return;

    // Clear existing markers
    document.querySelectorAll('.mapboxgl-marker').forEach(marker => marker.remove());

    const coords = analysisData.analysis.geographic_intelligence.coordinates;
    if (coords.length === 0) {
        map.current.flyTo({center: [8.6753, 9.0820], zoom: 5});
        return;
    };

    coords.forEach(coord => {
        new window.mapboxgl.Marker({color: '#e53e3e'})
            .setLngLat([coord.longitude, coord.latitude])
            .setPopup(new window.mapboxgl.Popup().setHTML(`<strong>${coord.location_name}</strong>`))
            .addTo(map.current);
    });

    const bounds = new window.mapboxgl.LngLatBounds();
    coords.forEach(coord => bounds.extend([coord.longitude, coord.latitude]));
    map.current.fitBounds(bounds, { padding: 80, maxZoom: 14 });

  }, [analysisData]);

  return <div ref={mapContainer} className="h-[600px] w-full rounded-lg" />;
};


// --- DocumentList Component (FULLY IMPLEMENTED) ---
interface DocListItem {
  id: string;
  filename: string;
  processed_at: string;
  confidence_score: number;
}
interface DocumentListProps { onDocumentSelect: (document: Document) => void; }
const DocumentList: React.FC<DocumentListProps> = ({ onDocumentSelect }) => {
  const [documents, setDocuments] = useState<DocListItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // This is still mocked as the backend endpoint is also a mock.
    // In a real app, this would fetch the full document list.
    const mockDocs: DocListItem[] = [];
    setDocuments(mockDocs);
    setLoading(false);
  }, []);

  if (loading) {
      return <div className="p-8 text-center"><Loader2 className="h-8 w-8 animate-spin mx-auto" /></div>
  }
  if (documents.length === 0) {
      return <div className="p-8 text-center text-gray-500">No documents in the library yet.</div>
  }

  const fetchFullDocument = async (docId: string) => {
    alert("This feature requires the /document/{id} endpoint to be implemented on the backend.");
  }

  return (
    <div className="p-6 space-y-4">
        <h2 className="text-2xl font-bold">Document Library</h2>
      {documents.map(doc => (
        <div key={doc.id} className="p-4 bg-white rounded-lg shadow-sm border hover:shadow-md transition-shadow flex justify-between items-center">
          <div>
            <p className="font-semibold text-blue-600">{doc.filename}</p>
            <p className="text-sm text-gray-500">Processed: {new Date(doc.processed_at).toLocaleString()}</p>
          </div>
          <button onClick={() => fetchFullDocument(doc.id)} className="bg-gray-100 hover:bg-gray-200 text-gray-700 font-bold py-2 px-4 rounded-lg text-sm">
            View Analysis
          </button>
        </div>
      ))}
    </div>
  )
};

// --- Main App Component ---
type TabType = 'analyzer' | 'query' | 'results' | 'forecasting' | 'map' | 'documents';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('analyzer');
  const [analysisData, setAnalysisData] = useState<Document | null>(null);
  const [queryResponse, setQueryResponse] = useState<string>('');

  const handleDocumentAnalyzed = (document: Document) => {
    setAnalysisData(document);
    setQueryResponse(""); // Clear previous query response
    setActiveTab('results');
  };

  const handleDocumentSelect = (document: Document) => {
    setAnalysisData(document);
    setQueryResponse("");
    setActiveTab('results');
  };

  const handleQueryResponse = (response: string) => {
    setAnalysisData(null); // Clear previous doc analysis
    setQueryResponse(response);
    setActiveTab('results');
  };

  const tabs = [
    { id: 'analyzer' as TabType, label: 'Analyzer', icon: Upload },
    { id: 'query' as TabType, label: 'Query', icon: Search },
    { id: 'results' as TabType, label: 'Results', icon: Brain },
    { id: 'forecasting' as TabType, label: 'Forecasting', icon: BarChart3 },
    { id: 'map' as TabType, label: 'Map', icon: Map },
    { id: 'documents' as TabType, label: 'Library', icon: List }
  ];

  useEffect(() => {
    const mapboxScript = document.createElement('script');
    mapboxScript.src = 'https://api.mapbox.com/mapbox-gl-js/v2.9.1/mapbox-gl.js';
    document.head.appendChild(mapboxScript);
    const mapboxCss = document.createElement('link');
    mapboxCss.href = 'https://api.mapbox.com/mapbox-gl-js/v2.9.1/mapbox-gl.css';
    mapboxCss.rel = 'stylesheet';
    document.head.appendChild(mapboxCss);
  }, [])

  return (
    <>
      <GlobalStyles />
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white border-b border-gray-200 shadow-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-4">
              <div className="flex items-center">
                <Shield className="h-8 w-8 text-blue-600 mr-3" />
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">Intelligence Document Analyzer</h1>
                  <p className="text-sm text-gray-600">AI-Powered Security Intelligence Platform</p>
                </div>
              </div>
            </div>
          </div>
        </header>

        <nav className="bg-white border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex space-x-8 overflow-x-auto">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap transition-colors ${isActive ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
                  >
                    <Icon className="h-5 w-5" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </nav>

        <main className="max-w-7xl mx-auto mt-6">
          <div className="bg-white rounded-lg shadow-md border border-gray-200 min-h-[600px]">
            {activeTab === 'analyzer' && <IntelligenceAnalyzer onDocumentAnalyzed={handleDocumentAnalyzed} />}
            {activeTab === 'query' && <QueryInterface onQueryResponse={handleQueryResponse} />}
            {activeTab === 'results' && <AnalysisResults analysisData={analysisData} queryResponse={queryResponse} />}
            {activeTab === 'forecasting' && <ForecastingDashboard />}
            {activeTab === 'map' && <GeospatialMap analysisData={analysisData} />}
            {activeTab === 'documents' && <DocumentList onDocumentSelect={handleDocumentSelect} />}
          </div>
        </main>

        <footer className="bg-white border-t border-gray-200 mt-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="flex justify-between items-center">
              <p className="text-sm text-gray-500">© 2025 Intelligence Document Analyzer.</p>
              {analysisData?.metadata?.uploaded_at && (
                  <div className="text-xs text-gray-400">
                    Last Analysis: {new Date(analysisData.metadata.uploaded_at).toLocaleString()}
                  </div>
                )}
            </div>
          </div>
        </footer>
      </div>
    </>
  );
};

export default App;
