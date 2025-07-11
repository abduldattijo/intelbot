// src/components/DocumentList.tsx - Enhanced Multi-Crime Document Management

import React, { useState, useEffect } from 'react';
import { FileText, Calendar, Eye, Trash2, Download, Search, Filter, AlertCircle, Loader2, BarChart3, Target, Database, RefreshCw } from 'lucide-react';

interface DocumentListProps {
  onDocumentSelect: (documentId: string) => void;
}

interface DocumentListItem {
  id: string;
  filename: string;
  file_type: string;
  processed_at: string;
  confidence_score: number;
  intelligence_summary: string;
}

interface CrimeTypeStats {
  crime_type: string;
  document_count: number;
  total_incidents: number;
  latest_date: string;
}

const DocumentList: React.FC<DocumentListProps> = ({ onDocumentSelect }) => {
  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [crimeTypeStats, setCrimeTypeStats] = useState<CrimeTypeStats[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'date' | 'filename' | 'confidence'>('date');
  const [filterType, setFilterType] = useState<string>('all');
  const [selectedCrimeType, setSelectedCrimeType] = useState<string>('all');
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [showStats, setShowStats] = useState(true);

  useEffect(() => {
    fetchDocuments();
    fetchCrimeTypeStats();
  }, []);

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/document-list');

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setDocuments(data.documents || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch documents');
    } finally {
      setLoading(false);
    }
  };

  const fetchCrimeTypeStats = async () => {
    try {
      // Fetch crime types and their stats
      const crimeTypesResponse = await fetch('http://localhost:8000/crime-types');
      if (crimeTypesResponse.ok) {
        const crimeTypesData = await crimeTypesResponse.json();

        // For each crime type, get some basic stats
        const statsPromises = crimeTypesData.crime_types.map(async (crimeType: string) => {
          try {
            const statsResponse = await fetch(`http://localhost:8000/monthly-chart-data?crime_type=${encodeURIComponent(crimeType)}`);
            if (statsResponse.ok) {
              const statsData = await statsResponse.json();
              return {
                crime_type: crimeType,
                document_count: statsData.summary?.total_months || 0,
                total_incidents: statsData.summary?.total_incidents || 0,
                latest_date: statsData.monthly_data?.[statsData.monthly_data.length - 1]?.date || 'N/A'
              };
            }
          } catch (err) {
            console.error(`Failed to fetch stats for ${crimeType}:`, err);
          }

          return {
            crime_type: crimeType,
            document_count: 0,
            total_incidents: 0,
            latest_date: 'N/A'
          };
        });

        const stats = await Promise.all(statsPromises);
        setCrimeTypeStats(stats.filter(s => s.document_count > 0));
      }
    } catch (err) {
      console.error('Failed to fetch crime type stats:', err);
    }
  };

  const deleteDocument = async (docId: string, filename: string) => {
    if (!confirm(`Are you sure you want to delete "${filename}"? This action cannot be undone and will affect crime statistics.`)) {
      return;
    }

    setDeletingId(docId);

    try {
      const response = await fetch(`http://localhost:8000/document/${docId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `Failed to delete document. Status: ${response.status}`);
      }

      setDocuments(prevDocuments => prevDocuments.filter(doc => doc.id !== docId));

      // Refresh crime type stats after deletion
      await fetchCrimeTypeStats();

      alert(`Successfully deleted "${filename}".`);

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
      console.error('Deletion failed:', errorMessage);
      alert(`Error: ${errorMessage}`);
    } finally {
      setDeletingId(null);
    }
  };

  const exportDocument = (doc: DocumentListItem) => {
    const exportData = {
      filename: doc.filename,
      processed_at: doc.processed_at,
      confidence_score: doc.confidence_score,
      intelligence_summary: doc.intelligence_summary,
      export_timestamp: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${doc.filename}_analysis.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportAllStats = () => {
    const exportData = {
      export_timestamp: new Date().toISOString(),
      total_documents: documents.length,
      crime_type_statistics: crimeTypeStats,
      document_summary: documents.map(doc => ({
        filename: doc.filename,
        file_type: doc.file_type,
        processed_at: doc.processed_at,
        confidence_score: doc.confidence_score
      }))
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `multi_crime_analysis_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Enhanced filtering logic
  const filteredDocuments = documents
    .filter(doc => {
      const matchesSearch = doc.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           doc.intelligence_summary.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesFileType = filterType === 'all' || doc.file_type === filterType;

      // Crime type filtering (basic - matches against filename/summary)
      let matchesCrimeType = true;
      if (selectedCrimeType !== 'all') {
        matchesCrimeType = doc.filename.toLowerCase().includes(selectedCrimeType.toLowerCase()) ||
                          doc.intelligence_summary.toLowerCase().includes(selectedCrimeType.toLowerCase());
      }

      return matchesSearch && matchesFileType && matchesCrimeType;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'date':
          return new Date(b.processed_at).getTime() - new Date(a.processed_at).getTime();
        case 'filename':
          return a.filename.localeCompare(b.filename);
        case 'confidence':
          return b.confidence_score - a.confidence_score;
        default:
          return 0;
      }
    });

  const getFileTypeIcon = (fileType: string) => {
    return <FileText className="h-5 w-5 text-blue-600" />;
  };

  const getConfidenceColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-50';
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  const uniqueFileTypes = [...new Set(documents.map(doc => doc.file_type))];
  const uniqueCrimeTypes = crimeTypeStats.map(stat => stat.crime_type);

  if (loading) {
    return (
      <div className="p-8">
        <div className="text-center">
          <div className="inline-flex items-center px-4 py-2 bg-blue-50 border border-blue-200 rounded-lg">
            <Loader2 className="animate-spin h-5 w-5 text-blue-600 mr-2" />
            <span className="text-blue-700 font-medium">Loading multi-crime document library...</span>
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
            <AlertCircle className="h-5 w-5 text-red-600 mr-2" />
            <h3 className="text-lg font-medium text-red-800">Error Loading Documents</h3>
          </div>
          <p className="text-red-700 mt-2">{error}</p>
          <button
            onClick={fetchDocuments}
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Retry Loading
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
            <FileText className="h-6 w-6 text-blue-600 mr-2" />
            Multi-Crime Document Library
          </h2>
          <p className="text-gray-600 mt-1">
            Manage and review your processed intelligence documents across all crime types
          </p>
        </div>

        <div className="flex items-center gap-4">
          <div className="text-right">
            <div className="text-2xl font-bold text-blue-600">{documents.length}</div>
            <div className="text-sm text-gray-500">Total Documents</div>
          </div>
          <button
            onClick={exportAllStats}
            className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
          >
            <Download className="h-4 w-4 mr-2" />
            Export All
          </button>
        </div>
      </div>

      {/* Crime Type Statistics */}
      {showStats && crimeTypeStats.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow border">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 flex items-center">
              <BarChart3 className="h-5 w-5 text-purple-600 mr-2" />
              Crime Type Overview
            </h3>
            <div className="flex gap-2">
              <button
                onClick={fetchCrimeTypeStats}
                className="px-3 py-1 text-sm bg-purple-100 text-purple-700 rounded hover:bg-purple-200"
              >
                <RefreshCw className="h-4 w-4 inline mr-1" />
                Refresh
              </button>
              <button
                onClick={() => setShowStats(false)}
                className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
              >
                Hide
              </button>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {crimeTypeStats.map((stat, index) => (
              <div key={stat.crime_type} className="bg-gray-50 p-4 rounded-lg border">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-gray-900 text-sm">{stat.crime_type}</h4>
                  <Target className="h-4 w-4 text-blue-600" />
                </div>
                <div className="space-y-1 text-sm text-gray-600">
                  <div className="flex justify-between">
                    <span>Documents:</span>
                    <span className="font-semibold">{stat.document_count}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Total Incidents:</span>
                    <span className="font-semibold text-red-600">{stat.total_incidents.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Latest:</span>
                    <span className="font-semibold">{stat.latest_date !== 'N/A' ? new Date(stat.latest_date).toLocaleDateString() : 'N/A'}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {!showStats && crimeTypeStats.length > 0 && (
        <div className="text-center">
          <button
            onClick={() => setShowStats(true)}
            className="px-4 py-2 bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 flex items-center mx-auto"
          >
            <BarChart3 className="h-4 w-4 mr-2" />
            Show Crime Type Statistics
          </button>
        </div>
      )}

      {/* Search and Filters */}
      <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
        <div className="flex flex-col md:flex-row gap-4">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search documents by filename or content..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          {/* Crime Type Filter */}
          <select
            value={selectedCrimeType}
            onChange={(e) => setSelectedCrimeType(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="all">All Crime Types</option>
            {uniqueCrimeTypes.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>

          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="date">Sort by Date</option>
            <option value="filename">Sort by Filename</option>
            <option value="confidence">Sort by Confidence</option>
          </select>

          {/* File Type Filter */}
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="all">All File Types</option>
            {uniqueFileTypes.map(type => (
              <option key={type} value={type}>{type.toUpperCase()}</option>
            ))}
          </select>
        </div>

        {/* Active Filters Display */}
        {(searchTerm || selectedCrimeType !== 'all' || filterType !== 'all') && (
          <div className="mt-3 flex flex-wrap gap-2">
            {searchTerm && (
              <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                Search: "{searchTerm}"
              </span>
            )}
            {selectedCrimeType !== 'all' && (
              <span className="px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded-full">
                Crime: {selectedCrimeType}
              </span>
            )}
            {filterType !== 'all' && (
              <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">
                Type: {filterType.toUpperCase()}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Document List */}
      {filteredDocuments.length === 0 ? (
        <div className="text-center bg-gray-50 border border-gray-200 rounded-lg p-8">
          <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            {documents.length === 0 ? 'No Documents' : 'No Matching Documents'}
          </h3>
          <p className="text-gray-600">
            {documents.length === 0
              ? 'Upload your first multi-crime intelligence document to get started.'
              : 'Try adjusting your search terms or filters.'
            }
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          {filteredDocuments.map((doc) => (
            <div key={doc.id} className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-4 flex-1">
                  <div className="flex-shrink-0">
                    {getFileTypeIcon(doc.file_type)}
                  </div>

                  <div className="flex-1 min-w-0">
                    <h3 className="text-lg font-medium text-gray-900 truncate">
                      {doc.filename}
                    </h3>

                    <div className="flex items-center space-x-4 mt-2">
                      <div className="flex items-center text-sm text-gray-500">
                        <Calendar className="h-4 w-4 mr-1" />
                        {new Date(doc.processed_at).toLocaleDateString()}
                      </div>

                      <span className="text-sm text-gray-500 uppercase">
                        {doc.file_type}
                      </span>

                      <span className={`px-2 py-1 rounded text-xs font-medium ${getConfidenceColor(doc.confidence_score)}`}>
                        {(doc.confidence_score * 100).toFixed(1)}% Confidence
                      </span>

                      {/* Crime Type Badge */}
                      {uniqueCrimeTypes.find(type =>
                        doc.filename.toLowerCase().includes(type.toLowerCase()) ||
                        doc.intelligence_summary.toLowerCase().includes(type.toLowerCase())
                      ) && (
                        <span className="px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded-full">
                          {uniqueCrimeTypes.find(type =>
                            doc.filename.toLowerCase().includes(type.toLowerCase()) ||
                            doc.intelligence_summary.toLowerCase().includes(type.toLowerCase())
                          )}
                        </span>
                      )}
                    </div>

                    <p className="text-gray-600 mt-2 text-sm line-clamp-2">
                      {doc.intelligence_summary}
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-2 ml-4">
                  <button
                    onClick={() => onDocumentSelect(doc.id)}
                    className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                    title="View Details"
                    disabled={deletingId === doc.id}
                  >
                    <Eye className="h-4 w-4" />
                  </button>

                  <button
                    onClick={() => exportDocument(doc)}
                    className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                    title="Export"
                    disabled={deletingId === doc.id}
                  >
                    <Download className="h-4 w-4" />
                  </button>

                  <button
                    onClick={() => deleteDocument(doc.id, doc.filename)}
                    className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors disabled:text-gray-400"
                    title="Delete"
                    disabled={deletingId === doc.id}
                  >
                    {deletingId === doc.id ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Enhanced Library Statistics */}
      {documents.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Database className="h-5 w-5 text-blue-600 mr-2" />
            Multi-Crime Library Statistics
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{documents.length}</div>
              <div className="text-sm text-gray-500">Total Documents</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{crimeTypeStats.length}</div>
              <div className="text-sm text-gray-500">Crime Types</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {documents.filter(doc => doc.confidence_score >= 0.8).length}
              </div>
              <div className="text-sm text-gray-500">High Confidence</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-600">
                {documents.filter(doc => doc.confidence_score >= 0.6 && doc.confidence_score < 0.8).length}
              </div>
              <div className="text-sm text-gray-500">Medium Confidence</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">
                {crimeTypeStats.reduce((sum, stat) => sum + stat.total_incidents, 0).toLocaleString()}
              </div>
              <div className="text-sm text-gray-500">Total Incidents</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DocumentList;