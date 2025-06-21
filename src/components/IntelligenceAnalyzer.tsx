import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area } from 'recharts';
import { TrendingUp, MapPin, AlertTriangle, Target, Calendar, Users, Shield, Brain, FileText, Activity, Upload, CheckCircle } from 'lucide-react';

interface ChatMessage {
  type: 'user' | 'analyst';
  content: string;
  timestamp: string;
}

interface UploadedFile {
  id: string;
  name: string;
  uploadedAt: string;
  analysis: {
    confidence_score: number;
    intelligence_summary: string;
    document_classification: { primary_type: string };
    geographic_intelligence: { total_locations: number };
    sentiment_analysis: { threat_level: string };
  };
}

interface AnalysisResult {
  document_id: string;
  analysis: {
    confidence_score: number;
    intelligence_summary: string;
    document_classification: { primary_type: string };
    geographic_intelligence: { total_locations: number };
    sentiment_analysis: { threat_level: string };
  };
}

interface AnalysisCardProps {
  title: string;
  children: React.ReactNode;
  icon: React.ComponentType<{ className?: string }>;
  classification?: string;
}

const IntelligenceAnalyzer = () => {
  const [selectedAnalysis, setSelectedAnalysis] = useState<string>('upload');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [userQuery, setUserQuery] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);

  // Sample intelligence data
  const monthlyData = [
    { month: 'Jan', incidents: 632, casualties: 331, arrests: 1103, armedRobbery: 309 },
    { month: 'Feb', incidents: 578, casualties: 228, arrests: 775, armedRobbery: 252 },
    { month: 'Mar', incidents: 700, casualties: 533, arrests: 959, armedRobbery: 265 },
    { month: 'Apr', incidents: 568, casualties: 322, arrests: 517, armedRobbery: 228 },
    { month: 'May', incidents: 500, casualties: 270, arrests: 585, armedRobbery: 176 },
    { month: 'Jun', incidents: 505, casualties: 260, arrests: 650, armedRobbery: 200 }
  ];

  const zoneData = [
    { zone: 'North-West', percentage: 44, incidents: 2844 },
    { zone: 'North-Central', percentage: 18, incidents: 1165 },
    { zone: 'South-South', percentage: 13, incidents: 849 },
    { zone: 'South-East', percentage: 11, incidents: 712 },
    { zone: 'South-West', percentage: 9, incidents: 583 },
    { zone: 'North-East', percentage: 5, indices: 324 }
  ];

  const crimeData = [
    { type: 'Armed Robbery', value: 2739, color: '#dc2626' },
    { type: 'Murder', value: 1896, color: '#7c2d12' },
    { type: 'Cattle Rustling', value: 1322, color: '#ea580c' },
    { type: 'Kidnapping', value: 352, color: '#be185d' }
  ];

  // Document upload functionality
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setAnalysisResult(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('analysis_type', 'full');

    try {
      const response = await fetch('http://localhost:8000/upload-document', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result: AnalysisResult = await response.json();
      setAnalysisResult(result);
      setUploadedFiles(prev => [...prev, {
        id: result.document_id,
        name: selectedFile.name,
        uploadedAt: new Date().toLocaleString(),
        analysis: result.analysis
      }]);
      setSelectedFile(null);
    } catch (err) {
      console.error('Upload error:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      alert(`Upload failed: ${errorMessage}`);
    } finally {
      setUploading(false);
    }
  };

  // NLP Query Processing
  const processQuery = (query: string): string => {
    const lowerQuery = query.toLowerCase();

    if (lowerQuery.includes('trend') && lowerQuery.includes('robbery')) {
      return "**Armed Robbery Trend Analysis**: Peak incidents in January (309), lowest in May (176). Average 238 incidents/month. Seasonal pattern shows increase during festive periods.";
    } else if (lowerQuery.includes('geographic') || lowerQuery.includes('location')) {
      return "**Geographic Analysis**: North-West zone dominates with 44% of incidents (2,844 cases). Primary hotspots: Zamfara, Katsina, Kaduna states. Rural-urban transition zones most affected.";
    } else if (lowerQuery.includes('forecast') || lowerQuery.includes('predict')) {
      return "**Threat Forecast**: Q2 2024 prediction shows 15-20% potential increase. High-risk periods: July-August. Focus areas: North-West zone continuation, urban expansion likely.";
    } else {
      return "**Intelligence Query Assistant**: I can analyze trends, geographic patterns, forecasting, crime types, and uploaded document intelligence. Ask specific questions about security patterns.";
    }
  };

  const handleQuery = () => {
    if (!userQuery.trim()) return;

    setIsProcessing(true);
    setTimeout(() => {
      const response = processQuery(userQuery);
      setChatMessages(prev => [...prev,
        { type: 'user' as const, content: userQuery, timestamp: new Date().toLocaleTimeString() },
        { type: 'analyst' as const, content: response, timestamp: new Date().toLocaleTimeString() }
      ]);
      setUserQuery('');
      setIsProcessing(false);
    }, 1000);
  };

  const AnalysisCard: React.FC<AnalysisCardProps> = ({ title, children, icon: Icon, classification = "RESTRICTED" }) => (
    <div className="bg-white rounded-lg shadow-lg p-6 mb-6 border-l-4 border-blue-600">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <Icon className="h-6 w-6 text-blue-600 mr-2" />
          <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
        </div>
        <span className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded">{classification}</span>
      </div>
      {children}
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-slate-900 text-white p-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center">
            <Shield className="h-8 w-8 mr-3" />
            <div>
              <h1 className="text-2xl font-bold">Intelligence Document Processor</h1>
              <p className="text-sm text-gray-300">Advanced AI-Powered Intelligence Analysis System</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm bg-red-600 px-3 py-1 rounded mb-1">SECRET//NOFORN</div>
            <div className="text-xs text-gray-300">For Official Use Only</div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-6">
        {/* Navigation */}
        <div className="mb-6">
          <nav className="flex flex-wrap gap-2">
            {[
              { id: 'upload', label: 'Document Upload', icon: Upload },
              { id: 'dashboard', label: 'Intelligence Dashboard', icon: Activity },
              { id: 'geographic', label: 'Geospatial Analysis', icon: MapPin },
              { id: 'nlp', label: 'NLP Assistant', icon: Brain },
              { id: 'forecast', label: 'Threat Forecasting', icon: Calendar }
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setSelectedAnalysis(id)}
                className={`flex items-center px-4 py-2 rounded-lg font-medium transition-colors ${
                  selectedAnalysis === id
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                }`}
              >
                <Icon className="h-4 w-4 mr-2" />
                {label}
              </button>
            ))}
          </nav>
        </div>

        {/* Document Upload Section */}
        {selectedAnalysis === 'upload' && (
          <div>
            <AnalysisCard title="Document Intelligence Processor" icon={Upload} classification="SECRET">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors mb-6">
                <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <div className="space-y-2">
                  <p className="text-lg font-medium text-gray-900">Upload Intelligence Documents</p>
                  <p className="text-gray-500">Supports PDF, Word, Text, Images with OCR</p>
                </div>
                <input
                  type="file"
                  accept=".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png,.tiff,.bmp"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="mt-4 inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 cursor-pointer transition-colors"
                >
                  Choose Document
                </label>
              </div>

              {selectedFile && (
                <div className="mb-6 p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <FileText className="h-8 w-8 text-blue-600" />
                      <div>
                        <p className="font-medium text-gray-900">{selectedFile.name}</p>
                        <p className="text-sm text-gray-500">
                          {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={handleUpload}
                      disabled={uploading}
                      className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
                    >
                      <Brain className="h-4 w-4" />
                      <span>{uploading ? 'Processing...' : 'Analyze Document'}</span>
                    </button>
                  </div>
                </div>
              )}

              {uploading && (
                <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                    <p className="text-blue-800">Processing document with AI intelligence extraction...</p>
                  </div>
                </div>
              )}

              {analysisResult && (
                <div className="space-y-6">
                  <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <CheckCircle className="h-6 w-6 text-green-600" />
                      <p className="text-green-800 font-medium">Document processed successfully!</p>
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6 border border-blue-200">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                      <Brain className="h-5 w-5 text-blue-600 mr-2" />
                      Intelligence Summary
                    </h3>
                    <p className="text-gray-700 mb-4 text-lg">{analysisResult.analysis.intelligence_summary}</p>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                      <div className="bg-white p-4 rounded-lg border shadow-sm">
                        <p className="text-sm text-gray-600">Confidence Score</p>
                        <p className="text-2xl font-bold text-blue-600">
                          {(analysisResult.analysis.confidence_score * 100).toFixed(0)}%
                        </p>
                      </div>
                      <div className="bg-white p-4 rounded-lg border shadow-sm">
                        <p className="text-sm text-gray-600">Document Type</p>
                        <p className="text-lg font-semibold text-gray-900">
                          {analysisResult.analysis.document_classification.primary_type.replace('_', ' ').toUpperCase()}
                        </p>
                      </div>
                      <div className="bg-white p-4 rounded-lg border shadow-sm">
                        <p className="text-sm text-gray-600">Locations Found</p>
                        <p className="text-2xl font-bold text-green-600">
                          {analysisResult.analysis.geographic_intelligence.total_locations}
                        </p>
                      </div>
                      <div className="bg-white p-4 rounded-lg border shadow-sm">
                        <p className="text-sm text-gray-600">Threat Level</p>
                        <p className={`text-lg font-semibold px-2 py-1 rounded ${
                          analysisResult.analysis.sentiment_analysis.threat_level === 'High' ? 'bg-red-100 text-red-800' :
                          analysisResult.analysis.sentiment_analysis.threat_level === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-green-100 text-green-800'
                        }`}>
                          {analysisResult.analysis.sentiment_analysis.threat_level}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Uploaded Files History */}
              {uploadedFiles.length > 0 && (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold mb-4">Recently Processed Documents</h3>
                  <div className="space-y-2">
                    {uploadedFiles.slice(-5).map((file, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded border">
                        <div>
                          <p className="font-medium">{file.name}</p>
                          <p className="text-sm text-gray-500">{file.uploadedAt}</p>
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-medium text-green-600">Processed</p>
                          <p className="text-xs text-gray-500">
                            Confidence: {(file.analysis.confidence_score * 100).toFixed(0)}%
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </AnalysisCard>
          </div>
        )}

        {/* Intelligence Dashboard */}
        {selectedAnalysis === 'dashboard' && (
          <div>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
              <div className="bg-gradient-to-r from-red-600 to-red-700 text-white p-6 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-red-100">Total Incidents</p>
                    <p className="text-3xl font-bold">3,483</p>
                    <p className="text-sm text-red-200">+23% vs last period</p>
                  </div>
                  <AlertTriangle className="h-12 w-12 text-red-200" />
                </div>
              </div>
              <div className="bg-gradient-to-r from-orange-600 to-orange-700 text-white p-6 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-orange-100">Total Casualties</p>
                    <p className="text-3xl font-bold">1,944</p>
                    <p className="text-sm text-orange-200">85% Civilians</p>
                  </div>
                  <Users className="h-12 w-12 text-orange-200" />
                </div>
              </div>
              <div className="bg-gradient-to-r from-green-600 to-green-700 text-white p-6 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-green-100">Arrests Made</p>
                    <p className="text-3xl font-bold">4,589</p>
                    <p className="text-sm text-green-200">132% Success Rate</p>
                  </div>
                  <Shield className="h-12 w-12 text-green-200" />
                </div>
              </div>
              <div className="bg-gradient-to-r from-purple-600 to-purple-700 text-white p-6 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-purple-100">Documents Processed</p>
                    <p className="text-3xl font-bold">{uploadedFiles.length}</p>
                    <p className="text-sm text-purple-200">AI Analysis Complete</p>
                  </div>
                  <FileText className="h-12 w-12 text-purple-200" />
                </div>
              </div>
            </div>

            <AnalysisCard title="Monthly Intelligence Trends" icon={TrendingUp}>
              <LineChart width={900} height={400} data={monthlyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="incidents" stroke="#dc2626" strokeWidth={3} name="Incidents" />
                <Line type="monotone" dataKey="casualties" stroke="#ea580c" strokeWidth={2} name="Casualties" />
                <Line type="monotone" dataKey="arrests" stroke="#16a34a" strokeWidth={2} name="Arrests" />
              </LineChart>
            </AnalysisCard>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <AnalysisCard title="Crime Classification" icon={Target}>
                <PieChart width={400} height={300}>
                  <Pie
                    data={crimeData}
                    cx={200}
                    cy={150}
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${value}`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {crimeData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </AnalysisCard>

              <AnalysisCard title="Geographic Distribution" icon={MapPin}>
                <BarChart width={400} height={300} data={zoneData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="zone" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="percentage" fill="#3b82f6" />
                </BarChart>
              </AnalysisCard>
            </div>
          </div>
        )}

        {/* NLP Assistant */}
        {selectedAnalysis === 'nlp' && (
          <div>
            <AnalysisCard title="AI Intelligence Assistant" icon={Brain} classification="SECRET">
              <div className="h-96 overflow-y-auto bg-gray-50 rounded p-4 space-y-3 mb-4">
                {chatMessages.length === 0 && (
                  <div className="text-gray-500 text-center py-8">
                    <Brain className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                    <p className="text-lg font-medium">Intelligence Analysis Ready</p>
                    <p>Ask about trends, geographic patterns, forecasts, or document analysis.</p>
                  </div>
                )}
                {chatMessages.map((msg, index) => (
                  <div key={index} className={`p-4 rounded-lg ${
                    msg.type === 'user' ? 'bg-blue-100 ml-8' : 'bg-white mr-8 border-l-4 border-blue-500'
                  }`}>
                    <div className="flex justify-between items-center mb-2">
                      <div className="text-sm font-medium text-gray-600">
                        {msg.type === 'user' ? '👤 Intelligence Officer' : '🤖 AI Analyst'}
                      </div>
                      <div className="text-xs text-gray-500">{msg.timestamp}</div>
                    </div>
                    <div className="whitespace-pre-wrap text-sm">{msg.content}</div>
                  </div>
                ))}
                {isProcessing && (
                  <div className="flex items-center space-x-2 text-gray-500 bg-white p-4 rounded-lg mr-8">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                    <span>Analyzing intelligence data...</span>
                  </div>
                )}
              </div>

              <div className="flex space-x-2">
                <input
                  type="text"
                  value={userQuery}
                  onChange={(e) => setUserQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleQuery()}
                  placeholder="Ask about trends, patterns, forecasts, uploaded documents..."
                  className="flex-1 p-3 border rounded-lg"
                />
                <button
                  onClick={handleQuery}
                  disabled={isProcessing || !userQuery.trim()}
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
                >
                  Query
                </button>
              </div>
            </AnalysisCard>
          </div>
        )}

        {/* Geographic Analysis */}
        {selectedAnalysis === 'geographic' && (
          <div>
            <AnalysisCard title="Interactive Security Hotspot Map" icon={MapPin}>
              <div className="bg-gray-50 rounded-lg p-6 mb-6">
                <div className="relative">
                  {/* Nigeria Map SVG */}
                  <svg
                    viewBox="0 0 800 600"
                    className="w-full h-96 border rounded bg-blue-50"
                    style={{ maxHeight: '400px' }}
                  >
                    {/* Nigeria outline (simplified) */}
                    <path
                      d="M 150 200 L 200 180 L 280 160 L 350 170 L 420 180 L 480 200 L 520 240 L 550 280 L 580 320 L 590 380 L 580 420 L 560 450 L 520 480 L 480 500 L 420 510 L 350 520 L 280 510 L 220 500 L 180 480 L 150 450 L 130 400 L 120 350 L 130 300 L 140 250 Z"
                      fill="#e5e7eb"
                      stroke="#6b7280"
                      strokeWidth="2"
                    />

                    {/* State boundaries (simplified) */}
                    <g stroke="#9ca3af" strokeWidth="1" fill="none">
                      <line x1="200" y1="200" x2="350" y2="300" />
                      <line x1="300" y1="180" x2="400" y2="350" />
                      <line x1="400" y1="200" x2="500" y2="400" />
                      <line x1="200" y1="350" x2="450" y2="350" />
                      <line x1="300" y1="400" x2="500" y2="300" />
                    </g>

                    {/* Hotspot Markers */}
                    {[
                      { name: 'Zamfara', x: 280, y: 240, risk: 'Critical', incidents: 1157, crime: 'Cattle Rustling' },
                      { name: 'Katsina', x: 320, y: 200, risk: 'High', incidents: 775, crime: 'Armed Robbery' },
                      { name: 'Kaduna', x: 350, y: 280, risk: 'High', incidents: 474, crime: 'Kidnapping' },
                      { name: 'Plateau', x: 420, y: 320, risk: 'Medium', incidents: 303, crime: 'Ethnic Conflict' },
                      { name: 'Niger', x: 300, y: 320, risk: 'Medium', incidents: 199, crime: 'Highway Robbery' },
                      { name: 'Lagos', x: 220, y: 450, risk: 'Low', incidents: 89, crime: 'Urban Crime' },
                      { name: 'Abuja', x: 350, y: 350, risk: 'Medium', incidents: 156, crime: 'Various' },
                      { name: 'Borno', x: 520, y: 240, risk: 'High', incidents: 380, crime: 'Insurgency' }
                    ].map((hotspot, index) => (
                      <g key={index}>
                        {/* Hotspot Circle */}
                        <circle
                          cx={hotspot.x}
                          cy={hotspot.y}
                          r={hotspot.risk === 'Critical' ? 16 : hotspot.risk === 'High' ? 12 : hotspot.risk === 'Medium' ? 8 : 6}
                          fill={
                            hotspot.risk === 'Critical' ? '#dc2626' :
                            hotspot.risk === 'High' ? '#ea580c' :
                            hotspot.risk === 'Medium' ? '#d97706' : '#16a34a'
                          }
                          stroke="white"
                          strokeWidth="2"
                          className="cursor-pointer hover:opacity-80 transition-opacity"
                          onClick={() => alert(`${hotspot.name} State\nRisk: ${hotspot.risk}\nIncidents: ${hotspot.incidents}\nPrimary Crime: ${hotspot.crime}`)}
                        />

                        {/* Pulsing effect for Critical areas */}
                        {hotspot.risk === 'Critical' && (
                          <circle
                            cx={hotspot.x}
                            cy={hotspot.y}
                            r="20"
                            fill="none"
                            stroke="#dc2626"
                            strokeWidth="2"
                            opacity="0.6"
                          >
                            <animate
                              attributeName="r"
                              values="16;24;16"
                              dur="2s"
                              repeatCount="indefinite"
                            />
                            <animate
                              attributeName="opacity"
                              values="0.6;0.2;0.6"
                              dur="2s"
                              repeatCount="indefinite"
                            />
                          </circle>
                        )}

                        {/* State Labels */}
                        <text
                          x={hotspot.x}
                          y={hotspot.y + 30}
                          textAnchor="middle"
                          className="text-xs font-medium fill-gray-700"
                        >
                          {hotspot.name}
                        </text>

                        {/* Incident Count */}
                        <text
                          x={hotspot.x}
                          y={hotspot.y + 42}
                          textAnchor="middle"
                          className="text-xs fill-gray-600"
                        >
                          {hotspot.incidents}
                        </text>
                      </g>
                    ))}

                    {/* Geographic Zones (Background regions) */}
                    <g opacity="0.3">
                      {/* North-West */}
                      <polygon
                        points="200,180 350,170 350,300 220,280"
                        fill="#fecaca"
                        stroke="#dc2626"
                        strokeWidth="1"
                        strokeDasharray="5,5"
                      />
                      <text x="275" y="230" textAnchor="middle" className="text-sm font-medium fill-red-700">
                        North-West (44%)
                      </text>

                      {/* North-Central */}
                      <polygon
                        points="350,280 450,270 450,380 350,350"
                        fill="#fed7aa"
                        stroke="#ea580c"
                        strokeWidth="1"
                        strokeDasharray="5,5"
                      />
                      <text x="400" y="325" textAnchor="middle" className="text-sm font-medium fill-orange-700">
                        North-Central (18%)
                      </text>

                      {/* South regions */}
                      <polygon
                        points="200,400 400,380 400,500 200,480"
                        fill="#bbf7d0"
                        stroke="#16a34a"
                        strokeWidth="1"
                        strokeDasharray="5,5"
                      />
                      <text x="300" y="440" textAnchor="middle" className="text-sm font-medium fill-green-700">
                        Southern Zones (38%)
                      </text>
                    </g>

                    {/* Border Indicators */}
                    <g>
                      {/* Niger Border */}
                      <line x1="150" y1="200" x2="300" y2="160" stroke="#dc2626" strokeWidth="3" strokeDasharray="10,5" />
                      <text x="220" y="175" className="text-xs fill-red-600 font-medium">Niger Border (High Risk)</text>

                      {/* Chad Border */}
                      <line x1="450" y1="180" x2="580" y2="200" stroke="#ea580c" strokeWidth="3" strokeDasharray="10,5" />
                      <text x="510" y="195" className="text-xs fill-orange-600 font-medium">Chad Border (Medium Risk)</text>
                    </g>
                  </svg>
                </div>

                {/* Map Legend */}
                <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 bg-red-600 rounded-full"></div>
                    <span className="text-sm">Critical Risk</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 bg-orange-600 rounded-full"></div>
                    <span className="text-sm">High Risk</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 bg-yellow-600 rounded-full"></div>
                    <span className="text-sm">Medium Risk</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 bg-green-600 rounded-full"></div>
                    <span className="text-sm">Low Risk</span>
                  </div>
                </div>

                <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                  <p className="text-sm text-blue-800">
                    <strong>Interactive Map:</strong> Click on any hotspot marker to view detailed intelligence.
                    Larger circles indicate higher incident counts. Pulsing animation shows critical areas requiring immediate attention.
                  </p>
                </div>
              </div>
            </AnalysisCard>

            <AnalysisCard title="Geospatial Intelligence Analysis" icon={MapPin}>
              <div className="overflow-x-auto mb-6">
                <table className="min-w-full table-auto">
                  <thead>
                    <tr className="bg-gray-50">
                      <th className="px-4 py-2 text-left">State</th>
                      <th className="px-4 py-2 text-left">Risk Level</th>
                      <th className="px-4 py-2 text-left">Total Incidents</th>
                      <th className="px-4 py-2 text-left">Primary Crime</th>
                      <th className="px-4 py-2 text-left">Trend</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { state: 'Zamfara', risk: 'Critical', incidents: 1157, crime: 'Cattle Rustling', trend: '↗️' },
                      { state: 'Katsina', risk: 'High', incidents: 775, crime: 'Armed Robbery', trend: '↗️' },
                      { state: 'Kaduna', risk: 'High', incidents: 474, crime: 'Kidnapping', trend: '↗️' },
                      { state: 'Plateau', risk: 'Medium', incidents: 303, crime: 'Ethnic Conflict', trend: '↔️' },
                      { state: 'Niger', risk: 'Medium', incidents: 199, crime: 'Highway Robbery', trend: '↗️' },
                      { state: 'Lagos', risk: 'Low', incidents: 89, crime: 'Urban Crime', trend: '↘️' }
                    ].map((stateData, index) => (
                      <tr key={index} className="border-b hover:bg-gray-50">
                        <td className="px-4 py-2 font-medium">{stateData.state}</td>
                        <td className="px-4 py-2">
                          <span className={`px-2 py-1 rounded text-sm ${
                            stateData.risk === 'Critical' ? 'bg-red-100 text-red-800' :
                            stateData.risk === 'High' ? 'bg-orange-100 text-orange-800' :
                            stateData.risk === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-green-100 text-green-800'
                          }`}>
                            {stateData.risk}
                          </span>
                        </td>
                        <td className="px-4 py-2">{stateData.incidents}</td>
                        <td className="px-4 py-2">{stateData.crime}</td>
                        <td className="px-4 py-2 text-lg">{stateData.trend}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </AnalysisCard>

            <AnalysisCard title="Zone-Level Intelligence Distribution" icon={MapPin}>
              <BarChart width={900} height={400} data={zoneData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="zone" />
                <YAxis />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-white p-4 border rounded shadow-lg">
                          <p className="font-semibold">{label}</p>
                          <p>{`Percentage: ${data.percentage}%`}</p>
                          <p>{`Incidents: ${data.incidents}`}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="percentage" fill="#3b82f6" />
              </BarChart>
            </AnalysisCard>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <AnalysisCard title="Border Security Hotspots" icon={AlertTriangle}>
                <div className="space-y-4">
                  {[
                    { border: 'Nigeria-Niger', threat: 'High', issue: 'Arms smuggling, cattle rustling' },
                    { border: 'Nigeria-Chad', threat: 'Medium', issue: 'Insurgency spillover' },
                    { border: 'Nigeria-Cameroon', threat: 'Medium', issue: 'Cross-border raids' },
                    { border: 'Nigeria-Benin', threat: 'Low', issue: 'Smuggling activities' }
                  ].map((border, index) => (
                    <div key={index} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                      <div>
                        <p className="font-medium">{border.border}</p>
                        <p className="text-sm text-gray-600">{border.issue}</p>
                      </div>
                      <span className={`px-2 py-1 rounded text-sm ${
                        border.threat === 'High' ? 'bg-red-100 text-red-800' :
                        border.threat === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-green-100 text-green-800'
                      }`}>
                        {border.threat}
                      </span>
                    </div>
                  ))}
                </div>
              </AnalysisCard>

              <AnalysisCard title="Urban vs Rural Incidents" icon={Target}>
                <PieChart width={400} height={300}>
                  <Pie
                    data={[
                      { name: 'Rural Areas', value: 72, color: '#dc2626' },
                      { name: 'Urban Areas', value: 18, color: '#ea580c' },
                      { name: 'Highway/Transit', value: 10, color: '#3b82f6' }
                    ]}
                    cx={200}
                    cy={150}
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${value}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {[
                      { name: 'Rural Areas', value: 72, color: '#dc2626' },
                      { name: 'Urban Areas', value: 18, color: '#ea580c' },
                      { name: 'Highway/Transit', value: 10, color: '#3b82f6' }
                    ].map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </AnalysisCard>
            </div>
          </div>
        )}

        {/* Threat Forecasting */}
        {selectedAnalysis === 'forecast' && (
          <div>
            <AnalysisCard title="Threat Forecasting & Predictive Analysis" icon={Calendar}>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div className="bg-red-50 border border-red-200 rounded-lg p-6">
                  <h4 className="font-semibold text-red-800 mb-3 flex items-center">
                    <AlertTriangle className="h-5 w-5 mr-2" />
                    High Probability Scenarios
                  </h4>
                  <ul className="text-red-700 space-y-2 text-sm">
                    <li>• 15-20% increase in Q3 2024</li>
                    <li>• Rainy season cattle rustling spike</li>
                    <li>• North-West zone continued dominance</li>
                    <li>• Shift from cattle rustling to kidnapping</li>
                    <li>• Cross-border spillover from Niger</li>
                  </ul>
                </div>

                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
                  <h4 className="font-semibold text-yellow-800 mb-3 flex items-center">
                    <Calendar className="h-5 w-5 mr-2" />
                    Medium Probability Events
                  </h4>
                  <ul className="text-yellow-700 space-y-2 text-sm">
                    <li>• Urban area expansion of crimes</li>
                    <li>• Ethnic militia escalation</li>
                    <li>• Technology adoption by criminals</li>
                    <li>• Economic hardship driving recruitment</li>
                    <li>• Border closure impacts</li>
                  </ul>
                </div>

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                  <h4 className="font-semibold text-blue-800 mb-3 flex items-center">
                    <Shield className="h-5 w-5 mr-2" />
                    Intervention Impact Models
                  </h4>
                  <ul className="text-blue-700 space-y-2 text-sm">
                    <li>• Enhanced border security: 15-20% reduction</li>
                    <li>• Economic programs: 25% reduction</li>
                    <li>• Community policing: 30% reduction</li>
                    <li>• Combined approach: 45-50% reduction</li>
                    <li>• Technology deployment: 35% reduction</li>
                  </ul>
                </div>
              </div>

              <div className="bg-white border rounded-lg p-6 mb-6">
                <h4 className="font-semibold mb-4 flex items-center">
                  <TrendingUp className="h-5 w-5 mr-2" />
                  6-Month Incident Forecast
                </h4>
                <LineChart width={900} height={300} data={[
                  { month: 'Jul 2024', predicted: 580, confidence: 85 },
                  { month: 'Aug 2024', predicted: 620, confidence: 82 },
                  { month: 'Sep 2024', predicted: 595, confidence: 78 },
                  { month: 'Oct 2024', predicted: 540, confidence: 75 },
                  { month: 'Nov 2024', predicted: 670, confidence: 70 },
                  { month: 'Dec 2024', predicted: 720, confidence: 65 }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="predicted" stroke="#dc2626" strokeWidth={3} name="Predicted Incidents" />
                  <Line type="monotone" dataKey="confidence" stroke="#16a34a" strokeWidth={2} name="Confidence %" />
                </LineChart>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <AnalysisCard title="Seasonal Pattern Predictions" icon={Calendar}>
                  <div className="space-y-4">
                    {[
                      { season: 'Dry Season (Nov-Mar)', risk: 'High', pattern: 'Peak cattle rustling, easy mobility' },
                      { season: 'Rainy Season (Apr-Oct)', risk: 'Medium', pattern: 'Reduced rural access, urban shift' },
                      { season: 'Harmattan (Dec-Feb)', risk: 'Critical', pattern: 'Maximum criminal activity' },
                      { season: 'Planting Season (Mar-May)', risk: 'Medium', pattern: 'Farmer-herder conflicts' }
                    ].map((season, index) => (
                      <div key={index} className="p-4 border rounded-lg">
                        <div className="flex justify-between items-center mb-2">
                          <h5 className="font-medium">{season.season}</h5>
                          <span className={`px-2 py-1 rounded text-xs ${
                            season.risk === 'Critical' ? 'bg-red-100 text-red-800' :
                            season.risk === 'High' ? 'bg-orange-100 text-orange-800' :
                            'bg-yellow-100 text-yellow-800'
                          }`}>
                            {season.risk}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600">{season.pattern}</p>
                      </div>
                    ))}
                  </div>
                </AnalysisCard>

                <AnalysisCard title="Early Warning Indicators" icon={AlertTriangle}>
                  <div className="space-y-3">
                    {[
                      { indicator: 'Unusual livestock movement', status: 'Active', level: 'High' },
                      { indicator: 'Increased military deployment', status: 'Monitoring', level: 'Medium' },
                      { indicator: 'Ethnic tension reports', status: 'Active', level: 'High' },
                      { indicator: 'Cross-border activity', status: 'Active', level: 'Critical' },
                      { indicator: 'Economic stress indicators', status: 'Monitoring', level: 'Medium' }
                    ].map((item, index) => (
                      <div key={index} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                        <div>
                          <p className="font-medium text-sm">{item.indicator}</p>
                          <p className="text-xs text-gray-500">Status: {item.status}</p>
                        </div>
                        <span className={`px-2 py-1 rounded text-xs ${
                          item.level === 'Critical' ? 'bg-red-100 text-red-800' :
                          item.level === 'High' ? 'bg-orange-100 text-orange-800' :
                          'bg-yellow-100 text-yellow-800'
                        }`}>
                          {item.level}
                        </span>
                      </div>
                    ))}
                  </div>
                </AnalysisCard>
              </div>
            </AnalysisCard>
          </div>
        )}

        {/* Other sections would go here */}
      </div>
    </div>
  );
};

export default IntelligenceAnalyzer;