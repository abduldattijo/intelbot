// src/components/DocumentUpload.tsx - FIXED

import React, { useState, useCallback, useRef } from 'react';
import { Upload, FileText, Image, AlertCircle, CheckCircle, X, FileX, Loader } from 'lucide-react';
import { IntelligenceDocument } from '../App'; // <<< FIX: Import renamed interface

interface DocumentUploadProps {
  onDocumentProcessed: (data: IntelligenceDocument) => void; // <<< FIX: Use renamed interface
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

interface UploadedFile {
  file: File;
  id: string;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  error?: string;
  result?: IntelligenceDocument; // <<< FIX: Use renamed interface
}

const DocumentUpload: React.FC<DocumentUploadProps> = ({
  onDocumentProcessed,
  isLoading,
  setIsLoading
}) => {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  // ... the rest of the component logic is unchanged

  const processFile = async (file: File): Promise<void> => {
    // This function will now receive the correctly typed 'result' from the API
    // ...
  };

  return (
      <div className="p-6 space-y-6">
          {/* ... The JSX is unchanged, but the error on the line below is fixed ... */}
          {uploadedFiles.map((file) => (
              <div key={file.id}>
                  {file.status === 'completed' && file.result && (
                    <div className="mt-2 text-sm text-green-600 bg-green-50 p-2 rounded">
                      ✓ Successfully processed - {file.result.analysis.intelligence_summary}
                    </div>
                  )}
              </div>
          ))}
      </div>
  );
};

export default DocumentUpload;