import React, { useEffect, useMemo, useState, useRef } from 'react';
import {
  Upload,
  FileText,
  Settings,
  Play,
  Download,
  Activity,
  MessageSquare,
  Bot,
  Send,
  Paperclip,
  Brain,
  CheckCircle,
  AlertCircle,
  Cpu,
  ArrowRight,
} from 'lucide-react';

// --- Configuration ---
const API_URL = 'http://localhost:8000/api';

// --- Shared Components ---

const Button = ({
  children,
  onClick,
  variant = 'primary',
  disabled,
  className = '',
  icon: Icon,
}) => {
  const baseStyle =
    'flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all duration-200 transform active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed';
  const variants = {
    primary:
      'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-900/20',
    secondary:
      'bg-slate-700 hover:bg-slate-600 text-slate-200 border border-slate-600',
    glass:
      'bg-white/10 hover:bg-white/20 text-white backdrop-blur-sm border border-white/10',
    outline:
      'border-2 border-indigo-500/50 text-indigo-400 hover:bg-indigo-500/10',
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseStyle} ${variants[variant]} ${className}`}
    >
      {Icon && <Icon size={18} />}
      {children}
    </button>
  );
};

const Card = ({ children, className = '' }) => (
  <div
    className={`bg-slate-800/50 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 ${className}`}
  >
    {children}
  </div>
);

// --- Portal Page Component ---

const PortalPage = () => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [topK, setTopK] = useState(10);
  const [threshold, setThreshold] = useState(0.7);
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnnotating, setIsAnnotating] = useState(false);
  const fileInputRef = useRef(null);

  // Health Check
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        if (!data.reference_data_loaded || !data.embeddings_indexed) {
          setStatus({
            type: 'warning',
            message:
              'System initialization incomplete: Reference data or embeddings not loaded.',
          });
        }
      } catch (error) {
        setStatus({
          type: 'error',
          message: 'Connection Failed: Backend server is unreachable.',
        });
      }
    };
    checkHealth();
  }, []);

  const annotationStats = useMemo(() => {
    if (!results?.annotations?.length) return null;

    const annotationCounts = {};
    let totalConfidence = 0;

    results.annotations.forEach((cell) => {
      const annotation = cell.predicted_annotation;
      annotationCounts[annotation] = (annotationCounts[annotation] || 0) + 1;
      totalConfidence += cell.confidence_score;
    });

    return {
      totalCells: results.total_cells,
      uniqueAnnotations: Object.keys(annotationCounts).length,
      avgConfidence: totalConfidence / results.annotations.length,
    };
  }, [results]);

  const showStatus = (message, type) => {
    setStatus({ message, type });
    // Auto dismiss success messages
    if (type === 'success') {
      setTimeout(() => setStatus(null), 5000);
    }
  };

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0] || null;
    processFile(selectedFile);
  };

  const processFile = (selectedFile) => {
    if (selectedFile) {
      if (selectedFile.name.endsWith('.h5ad')) {
        setFile(selectedFile);
        setFileName(selectedFile.name);
        showStatus('File ready for analysis', 'info');
      } else {
        showStatus('Invalid file type. Please upload a .h5ad file.', 'error');
      }
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const droppedFile = event.dataTransfer.files?.[0] || null;
    processFile(droppedFile);
  };

  const annotateData = async () => {
    if (!file) {
      showStatus('Please select a .h5ad file first.', 'error');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('top_k', topK);
    formData.append('similarity_threshold', threshold);

    setIsAnnotating(true);
    showStatus('Processing data vectors...', 'info');

    try {
      const response = await fetch(`${API_URL}/annotate`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Annotation failed');
      }

      const data = await response.json();
      setResults(data);
      showStatus(
        `Analysis complete: ${data.total_cells} cells annotated.`,
        'success'
      );
    } catch (error) {
      showStatus(`Error: ${error.message}`, 'error');
    } finally {
      setIsAnnotating(false);
    }
  };

  const downloadResults = () => {
    if (!results) return;

    const headers = [
      'Cell ID',
      'Predicted Annotation',
      'Confidence Score',
      'Top Matches',
    ];
    const rows = results.annotations.map((cell) => [
      cell.cell_id,
      cell.predicted_annotation,
      cell.confidence_score.toFixed(4),
      cell.top_matches
        .map((match) => `${match.annotation}:${match.similarity.toFixed(4)}`)
        .join(';'),
    ]);

    const csvContent = [
      headers.join(','),
      ...rows.map((row) => row.map((value) => `"${value}"`).join(',')),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `annotations_${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500">
      {/* Header Section */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-cyan-400 flex items-center gap-3">
            <Brain className="text-indigo-400" size={32} />
            Brain Tumor Annotation
          </h1>
          <p className="text-slate-400 mt-2">
            Glioma single-cell RNA-seq analysis via vector embeddings
          </p>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Input & Controls */}
        <div className="lg:col-span-1 space-y-6">
          {/* Upload Card */}
          <Card
            className={`relative overflow-hidden transition-all duration-300 group ${
              isDragging ? 'border-indigo-500 bg-indigo-500/10' : ''
            }`}
          >
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className="text-center p-8 border-2 border-dashed border-slate-600 rounded-xl hover:border-indigo-500/50 transition-colors"
            >
              <div className="mb-4 flex justify-center">
                <div className="p-4 bg-slate-700/50 rounded-full group-hover:scale-110 transition-transform duration-300">
                  {file ? (
                    <FileText className="text-emerald-400" size={32} />
                  ) : (
                    <Upload className="text-indigo-400" size={32} />
                  )}
                </div>
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">
                {file ? 'File Selected' : 'Upload Data'}
              </h3>
              <p className="text-sm text-slate-400 mb-6">
                {fileName || 'Drag & drop .h5ad file here'}
              </p>

              <div className="relative">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".h5ad"
                  onChange={handleFileChange}
                  className="hidden"
                />
                <Button
                  variant="secondary"
                  className="w-full"
                  onClick={() => fileInputRef.current?.click()}
                >
                  Browse Files
                </Button>
              </div>
            </div>
          </Card>

          {/* Parameters Card */}
          <Card>
            <div className="flex items-center gap-2 mb-6">
              <Settings className="text-slate-400" size={20} />
              <h3 className="text-lg font-semibold text-white">
                Analysis Parameters
              </h3>
            </div>

            <div className="space-y-6">
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-300">Neighbors (Top K)</span>
                  <span className="text-indigo-400 font-mono">{topK}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="50"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
              </div>

              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-300">Similarity Threshold</span>
                  <span className="text-indigo-400 font-mono">{threshold}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
              </div>

              <Button
                onClick={annotateData}
                disabled={isAnnotating || !file}
                className="w-full"
                icon={isAnnotating ? Activity : Play}
              >
                {isAnnotating ? 'Processing...' : 'Run Annotation'}
              </Button>
            </div>
          </Card>

          {/* Status Alert */}
          {status && (
            <div
              className={`p-4 rounded-xl border flex items-start gap-3 ${
                status.type === 'error'
                  ? 'bg-red-500/10 border-red-500/20 text-red-200'
                  : status.type === 'success'
                  ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-200'
                  : 'bg-blue-500/10 border-blue-500/20 text-blue-200'
              }`}
            >
              {status.type === 'error' ? (
                <AlertCircle size={20} />
              ) : status.type === 'success' ? (
                <CheckCircle size={20} />
              ) : (
                <Activity size={20} />
              )}
              <p className="text-sm">{status.message}</p>
            </div>
          )}
        </div>

        {/* Right Column: Results */}
        <div className="lg:col-span-2">
          {!results ? (
            <div className="h-full min-h-[400px] flex flex-col items-center justify-center border-2 border-dashed border-slate-700 rounded-2xl bg-slate-800/20 text-slate-500">
              <Cpu size={48} className="mb-4 opacity-50" />
              <p className="text-lg">Results will appear here</p>
              <p className="text-sm opacity-60">
                Upload a file and run annotation to begin
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Stats Grid */}
              {annotationStats && (
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <Card className="text-center p-4">
                    <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                      Total Cells
                    </p>
                    <p className="text-3xl font-bold text-white">
                      {annotationStats.totalCells.toLocaleString()}
                    </p>
                  </Card>
                  <Card className="text-center p-4">
                    <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                      Unique Types
                    </p>
                    <p className="text-3xl font-bold text-indigo-400">
                      {annotationStats.uniqueAnnotations}
                    </p>
                  </Card>
                  <Card className="text-center p-4">
                    <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                      Avg Confidence
                    </p>
                    <p className="text-3xl font-bold text-emerald-400">
                      {(annotationStats.avgConfidence * 100).toFixed(1)}%
                    </p>
                  </Card>
                </div>
              )}

              {/* Data Table */}
              <Card className="overflow-hidden p-0">
                <div className="p-4 border-b border-slate-700/50 flex justify-between items-center bg-slate-800/50">
                  <h3 className="font-semibold text-white">
                    Annotation Results
                  </h3>
                  <Button
                    variant="secondary"
                    onClick={downloadResults}
                    icon={Download}
                    className="py-1.5 px-4 text-sm"
                  >
                    Export CSV
                  </Button>
                </div>

                <div className="overflow-x-auto max-h-[600px]">
                  <table className="w-full text-left text-sm text-slate-300">
                    <thead className="bg-slate-900/50 text-slate-400 sticky top-0 z-10">
                      <tr>
                        <th className="p-4 font-medium">Cell ID</th>
                        <th className="p-4 font-medium">Prediction</th>
                        <th className="p-4 font-medium">Confidence</th>
                        <th className="p-4 font-medium hidden sm:table-cell">
                          Top Matches
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-700/50">
                      {results.annotations.slice(0, 100).map((cell) => (
                        <tr
                          key={cell.cell_id}
                          className="hover:bg-slate-700/30 transition-colors"
                        >
                          <td className="p-4 font-mono text-xs text-slate-500">
                            {cell.cell_id}
                          </td>
                          <td className="p-4 font-medium text-white">
                            {cell.predicted_annotation}
                          </td>
                          <td className="p-4">
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 rounded-full"
                                  style={{
                                    width: `${cell.confidence_score * 100}%`,
                                  }}
                                />
                              </div>
                              <span className="text-xs">
                                {(cell.confidence_score * 100).toFixed(0)}%
                              </span>
                            </div>
                          </td>
                          <td className="p-4 text-xs text-slate-500 hidden sm:table-cell">
                            {cell.top_matches
                              .slice(0, 2)
                              .map((m) => m.annotation)
                              .join(', ')}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {results.annotations.length > 100 && (
                    <div className="p-4 text-center text-slate-500 text-sm bg-slate-800/30">
                      ...and {results.annotations.length - 100} more cells
                    </div>
                  )}
                </div>
              </Card>
            </div>
          )}
        </div>
      </div>

      {/* Floating Chat Button */}
      <a href="#/chat" className="fixed bottom-8 right-8 z-50">
        <button className="bg-gradient-to-r from-indigo-600 to-cyan-600 hover:from-indigo-500 hover:to-cyan-500 text-white p-4 rounded-full shadow-lg shadow-indigo-500/30 transition-all duration-300 hover:scale-110 flex items-center justify-center group">
          <Bot size={28} className="group-hover:rotate-12 transition-transform" />
        </button>
      </a>
    </div>
  );
};

// --- Chat Page Component ---

const ChatPage = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content:
        'Hello! I am your research assistant for glioma analysis. You can upload data directly here or ask me questions about cell types and markers.',
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSend = () => {
    if (!inputValue.trim()) return;

    const newUserMsg = { id: Date.now(), role: 'user', content: inputValue };
    setMessages((prev) => [...prev, newUserMsg]);
    setInputValue('');
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      const responses = [
        'Based on the expression profile, these cells likely belong to the oligodendrocyte lineage.',
        "I've analyzed the uploaded dataset. The tumor core shows high heterogeneity.",
        'Would you like to compare these results with the TCGA reference database?',
        'The confidence score for that cluster is 92%, suggesting a strong match.',
      ];
      const randomResponse =
        responses[Math.floor(Math.random() * responses.length)];

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          role: 'assistant',
          content: randomResponse,
        },
      ]);
      setIsTyping(false);
    }, 1500);
  };

  return (
    <div className="max-w-5xl mx-auto h-[calc(100vh-4rem)] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between py-4 mb-4 border-b border-slate-700/50">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <MessageSquare className="text-cyan-400" />
          Research Assistant
        </h2>
        <a href="#/">
          <Button
            variant="secondary"
            className="px-4 py-2 text-sm"
            icon={ArrowRight}
          >
            Back to Portal
          </Button>
        </a>
      </div>

      {/* Messages Area */}
      <Card className="flex-1 mb-4 flex flex-col overflow-hidden bg-slate-900/50 border-slate-700/50">
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${
                msg.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`max-w-[80%] rounded-2xl p-4 ${
                  msg.role === 'user'
                    ? 'bg-indigo-600 text-white rounded-tr-none shadow-lg shadow-indigo-900/20'
                    : 'bg-slate-800 text-slate-200 rounded-tl-none border border-slate-700'
                }`}
              >
                <p className="leading-relaxed">{msg.content}</p>
              </div>
            </div>
          ))}

          {isTyping && (
            <div className="flex justify-start">
              <div className="bg-slate-800 rounded-2xl rounded-tl-none p-4 border border-slate-700 flex gap-2 items-center">
                <span
                  className="w-2 h-2 bg-slate-500 rounded-full animate-bounce"
                  style={{ animationDelay: '0ms' }}
                />
                <span
                  className="w-2 h-2 bg-slate-500 rounded-full animate-bounce"
                  style={{ animationDelay: '150ms' }}
                />
                <span
                  className="w-2 h-2 bg-slate-500 rounded-full animate-bounce"
                  style={{ animationDelay: '300ms' }}
                />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </Card>

      {/* Input Area */}
      <div className="relative">
        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          placeholder="Ask about your data or upload a file..."
          className="w-full bg-slate-800/80 backdrop-blur-md text-white rounded-xl border border-slate-700 p-4 pr-32 focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none shadow-xl"
          rows="3"
        />
        <div className="absolute bottom-4 right-4 flex gap-2">
          <button className="p-2 text-slate-400 hover:text-white transition-colors">
            <Paperclip size={20} />
          </button>
          <button
            onClick={handleSend}
            disabled={!inputValue.trim()}
            className="bg-indigo-600 hover:bg-indigo-500 text-white p-2 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send size={20} />
          </button>
        </div>
      </div>

      <div className="mt-4 flex gap-2 overflow-x-auto pb-2">
        {[
          'Analyze tumor core',
          'Identify rare cell types',
          'Compare with control',
          'Export report',
        ].map((suggestion) => (
          <button
            key={suggestion}
            onClick={() => setInputValue(suggestion)}
            className="whitespace-nowrap px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-full text-sm text-slate-300 transition-colors"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
};

// --- Main App Component ---

const App = () => {
  const getRoute = () => {
    const hash = window.location.hash.replace('#', '');
    if (hash.startsWith('/chat')) return 'chat';
    return 'portal';
  };

  const [route, setRoute] = useState(getRoute());

  useEffect(() => {
    const handleHashChange = () => setRoute(getRoute());
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-indigo-500/30">
      {/* Background Ambience */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-20%] left-[-10%] w-[600px] h-[600px] bg-indigo-900/20 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[600px] h-[600px] bg-cyan-900/10 rounded-full blur-[120px]" />
      </div>

      <main className="relative z-10 p-6 md:p-8">
        {route === 'chat' ? <ChatPage /> : <PortalPage />}
      </main>
    </div>
  );
};

export default App;
