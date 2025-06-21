// src/components/DocumentUpload.tsx - Enhanced Document Upload Component

import React, { useState, useCallback, useRef } from 'react';
import { Upload, FileText, Image, AlertCircle, CheckCircle, X, FileX, Loader } from 'lucide-react';
import { Document } from '../App';

interface DocumentUploadProps {
  onDocumentProcessed: (data: Document) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

interface UploadedFile {
  file: File;
  id: string;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  error?: string;
  result?: Document;
}

const DocumentUpload: React.FC<DocumentUploadProps> = ({
  onDocumentProcessed,
  isLoading,
  setIsLoading
}) => {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const supportedFormats = [
    { type: 'PDF', extensions: ['.pdf'], icon: FileText, color: 'text-red-600' },
    { type: 'Word', extensions: ['.doc', '.docx'], icon: FileText, color: 'text-blue-600' },
    { type: 'Text', extensions: ['.txt'], icon: FileText, color: 'text-gray-600' },
    { type: 'Images', extensions: ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'], icon: Image, color: 'text-green-600' }
  ];

  const isValidFileType = (file: File): boolean => {
    const validExtensions = supportedFormats.flatMap(format => format.extensions);
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    return validExtensions.includes(fileExtension);
  };

  const generateFileId = (): string => {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
  };

  const updateFileStatus = (id: string, updates: Partial<UploadedFile>) => {
    setUploadedFiles(prev => prev.map(file =>
      file.id === id ? { ...file, ...updates } : file
    ));
  };

  const processFile = async (file: File): Promise<void> => {
    const fileId = generateFileId();

    const uploadedFile: UploadedFile = {
      file,
      id: fileId,
      status: 'pending',
      progress: 0
    };

    setUploadedFiles(prev => [...prev, uploadedFile]);

    try {
      updateFileStatus(fileId, { status: 'uploading', progress: 25 });

      const formData = new FormData();
      formData.append('file', file);
      formData.append('analysis_type', 'full');

      updateFileStatus(fileId, { status: 'processing', progress: 50 });
      setIsLoading(true);

      const response = await fetch('http://localhost:8000/upload-document', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      updateFileStatus(fileId, {
        status: 'completed',
        progress: 100,
        result
      });

      onDocumentProcessed(result);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      updateFileStatus(fileId, {
        status: 'error',
        progress: 0,
        error: errorMessage
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleFiles = async (files: FileList | File[]) => {
    const fileArray = Array.from(files);

    for (const file of fileArray) {
      if (!isValidFileType(file)) {
        alert(`File "${file.name}" is not a supported format. Please upload PDF, Word, Text, or Image files.`);
        continue;
      }

      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        alert(`File "${file.name}" is too large. Please upload files smaller than 10MB.`);
        continue;
      }

      await processFile(file);
    }
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  }, []);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files);
    }
  };

  const removeFile = (id: string) => {
    setUploadedFiles(prev => prev.filter(file => file.id !== id));
  };

  const retryFile = (id: string) => {
    const file = uploadedFiles.find(f => f.id === id);
    if (file) {
      processFile(file.file);
    }
  };

  const getStatusIcon = (status: UploadedFile['status']) => {
    switch (status) {
      case 'pending':
      case 'uploading':
      case 'processing':
        return <Loader className="h-4 w-4 animate-spin text-blue-600" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-600" />;
      default:
        return null;
    }
  };

  const getStatusText = (file: UploadedFile) => {
    switch (file.status) {
      case 'pending':
        return 'Waiting...';
      case 'uploading':
        return 'Uploading...';
      case 'processing':
        return 'Processing with AI...';
      case 'completed':
        return 'Analysis Complete';
      case 'error':
        return file.error || 'Error occurred';
      default:
        return '';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Upload Intelligence Documents
        </h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Upload security reports, incident documents, or images for AI-powered intelligence analysis.
          Our system extracts key information, identifies patterns, and provides actionable insights.
        </p>
      </div>

      {/* Supported Formats */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-blue-800 mb-3">Supported File Formats:</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {supportedFormats.map((format) => {
            const Icon = format.icon;
            return (
              <div key={format.type} className="flex items-center space-x-2">
                <Icon className={`h-4 w-4 ${format.color}`} />
                <div>
                  <div className="text-sm font-medium text-gray-700">{format.type}</div>
                  <div className="text-xs text-gray-500">{format.extensions.join(', ')}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Upload Area */}
      <div
        className={`
          relative border-2 border-dashed rounded-lg p-8 text-center transition-colors duration-200
          ${dragActive 
            ? 'border-blue-500 bg-blue-50' 
            : 'border-gray-300 hover:border-gray-400'
          }
          ${isLoading ? 'opacity-50 pointer-events-none' : ''}
        `}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png,.tiff,.bmp"
          onChange={handleFileInput}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={isLoading}
        />

        <Upload className={`mx-auto h-12 w-12 mb-4 ${dragActive ? 'text-blue-500' : 'text-gray-400'}`} />

        <h3 className="text-lg font-medium text-gray-900 mb-2">
          {dragActive ? 'Drop files here' : 'Upload Documents'}
        </h3>

        <p className="text-gray-600 mb-4">
          Drag and drop files here, or click to select files
        </p>

        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={isLoading}
          className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Upload className="h-4 w-4 mr-2" />
          Choose Files
        </button>

        <p className="text-xs text-gray-500 mt-2">
          Maximum file size: 10MB per file
        </p>
      </div>

      {/* Upload Progress */}
      {uploadedFiles.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-gray-900">Upload Progress</h3>
          {uploadedFiles.map((file) => (
            <div key={file.id} className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-3">
                  {getStatusIcon(file.status)}
                  <div>
                    <p className="text-sm font-medium text-gray-900">{file.file.name}</p>
                    <p className="text-xs text-gray-500">
                      {(file.file.size / 1024 / 1024).toFixed(1)} MB
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-600">{getStatusText(file)}</span>

                  {file.status === 'error' && (
                    <button
                      onClick={() => retryFile(file.id)}
                      className="text-blue-600 hover:text-blue-700 text-sm font-medium"
                    >
                      Retry
                    </button>
                  )}

                  {file.status !== 'processing' && file.status !== 'uploading' && (
                    <button
                      onClick={() => removeFile(file.id)}
                      className="text-red-600 hover:text-red-700"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  )}
                </div>
              </div>

              {(file.status === 'uploading' || file.status === 'processing') && (
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${file.progress}%` }}
                  ></div>
                </div>
              )}

              {file.status === 'error' && file.error && (
                <div className="mt-2 text-sm text-red-600 bg-red-50 p-2 rounded">
                  {file.error}
                </div>
              )}

              {file.status === 'completed' && file.result && (
                <div className="mt-2 text-sm text-green-600 bg-green-50 p-2 rounded">
                  ✓ Successfully processed - {file.result.analysis.intelligence_summary}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Tips */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-800 mb-2">💡 Tips for Best Results:</h3>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• Ensure documents contain clear text and numerical data</li>
          <li>• For scanned documents, use high-quality images for better OCR results</li>
          <li>• Include documents with dates and location information for temporal analysis</li>
          <li>• Upload multiple related documents to improve pattern recognition</li>
        </ul>
      </div>
    </div>
  );
};

export default DocumentUpload;