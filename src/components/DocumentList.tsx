// src/components/DocumentList.tsx - Document Management Component

import React, { useState, useEffect } from 'react';
import { FileText, Calendar, Eye, Trash2, Download, Search, Filter, AlertCircle } from 'lucide-react';
import { Document } from '../App';

interface DocumentListProps {
  onDocumentSelect: (document: Document) => void;
}

interface DocumentListItem {
  id: string;
  filename: string;
  file_type: string;
  processed_at: string;
  confidence_score: number;
  intelligence_summary: string;
}

const DocumentList: React.FC<DocumentListProps> = ({ onDocumentSelect }) => {
  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'date' | 'filename' | 'confidence'>('date');
  const [filterType, setFilterType] = useState<string>('all');

  useEffect(() => {
    fetchDocuments();
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

  const fetchDocumentDetails = async (docId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/document/${docId}`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      onDocumentSelect(data);
    } catch (err) {
      alert(`Failed to load document details: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  const deleteDocument = async (docId: string, filename: string) => {
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
      return;
    }

    // Note: Delete endpoint would need to be implemented in backend
    alert('Delete functionality would be implemented in the backend');
  };

  const exportDocument = (doc: DocumentListItem) => {
    const exportData = {
      filename: doc.filename,
      processed_at: doc.processed_at,
      confidence_score: doc.confidence_score,
      intelligence_summary: doc.intelligence_summary
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${doc.filename}_analysis.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const filteredDocuments = documents
    .filter(doc => {
      const matchesSearch = doc.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           doc.intelligence_summary.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesFilter = filterType === 'all' || doc.file_type === filterType;
      return matchesSearch && matchesFilter;
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

  if (loading) {
    return (
      <div className="p-8">
        <div className="text-center">
          <div className="inline-flex items-center px-4 py-2 bg-blue-50 border border-blue-200 rounded-lg">
            <FileText className="animate-pulse h-5 w-5 text-blue-600 mr-2" />
            <span className="text-blue-700 font-medium">Loading documents...</span>
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
            Document Library
          </h2>
          <p className="text-gray-600 mt-1">
            Manage and review your processed intelligence documents
          </p>
        </div>

        <div className="text-right">
          <div className="text-2xl font-bold text-blue-600">{documents.length}</div>
          <div className="text-sm text-gray-500">Total Documents</div>
        </div>
      </div>

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

          {/* Filter */}
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
              ? 'Upload your first document to get started with intelligence analysis.'
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
                    </div>

                    <p className="text-gray-600 mt-2 text-sm line-clamp-2">
                      {doc.intelligence_summary}
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-2 ml-4">
                  <button
                    onClick={() => fetchDocumentDetails(doc.id)}
                    className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                    title="View Details"
                  >
                    <Eye className="h-4 w-4" />
                  </button>

                  <button
                    onClick={() => exportDocument(doc)}
                    className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                    title="Export"
                  >
                    <Download className="h-4 w-4" />
                  </button>

                  <button
                    onClick={() => deleteDocument(doc.id, doc.filename)}
                    className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                    title="Delete"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Summary Stats */}
      {documents.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Library Statistics</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{documents.length}</div>
              <div className="text-sm text-gray-500">Total Documents</div>
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
                {documents.filter(doc => doc.confidence_score < 0.6).length}
              </div>
              <div className="text-sm text-gray-500">Low Confidence</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DocumentList;