import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertCircle,
  Bot,
  CheckCircle,
  Cpu,
  Download,
  FileText,
  Play,
  Settings,
  Upload,
} from 'lucide-react';
import Button from './ui/Button';
import Card from './ui/Card';
import ChatPage from './ChatPage';

const API_URL = 'http://localhost:8000/api';

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
  const [showChat, setShowChat] = useState(false);

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
      showStatus(`Analysis complete: ${data.total_cells} cells annotated.`, 'success');
    } catch (error) {
      showStatus(`Error: ${error.message}`, 'error');
    } finally {
      setIsAnnotating(false);
    }
  };

  const downloadResults = () => {
    if (!results) return;

    const headers = ['Cell ID', 'Predicted Annotation', 'Confidence Score', 'Top Matches'];
    const rows = results.annotations.map((cell) => [
      cell.cell_id,
      cell.predicted_annotation,
      cell.confidence_score.toFixed(4),
      cell.top_matches.map((match) => `${match.annotation}:${match.similarity.toFixed(4)}`).join(';'),
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
    <div
      className={`w-full max-w-7xl lg:max-w-none mx-auto animate-in fade-in duration-500 py-8 px-4 md:px-8 ${
        showChat ? 'lg:pr-[640px]' : ''
      }`}
    >
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold text-white">Analysis Dashboard</h2>
          <p className="text-slate-400">Manage your datasets and run annotations</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mt-8">
        <div className="lg:col-span-1 space-y-6">
          <Card
            className={`relative overflow-hidden transition-all duration-300 group ${
              isDragging ? 'border-indigo-500 bg-indigo-500/10' : ''
            }`}
          >
            <div
              onDragOver={(event) => {
                event.preventDefault();
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
                  type="file"
                  accept=".h5ad"
                  onChange={handleFileChange}
                  ref={fileInputRef}
                  className="hidden"
                />
                <Button
                  variant="secondary"
                  className="w-full"
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                >
                  Browse Files
                </Button>
              </div>
            </div>
          </Card>

          <Card>
            <div className="flex items-center gap-2 mb-6">
              <Settings className="text-slate-400" size={20} />
              <h3 className="text-lg font-semibold text-white">Parameters</h3>
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
                  onChange={(event) => setTopK(Number(event.target.value))}
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
                  onChange={(event) => setThreshold(Number(event.target.value))}
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

        <div className="lg:col-span-2">
          {!results ? (
            <div className="h-full min-h-[400px] flex flex-col items-center justify-center border-2 border-dashed border-slate-700 rounded-2xl bg-slate-800/20 text-slate-500">
              <Cpu size={48} className="mb-4 opacity-50" />
              <p className="text-lg">Results will appear here</p>
              <p className="text-sm opacity-60">Upload a file and run annotation to begin</p>
            </div>
          ) : (
            <div className="space-y-6">
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

              <Card className="overflow-hidden p-0">
                <div className="p-4 border-b border-slate-700/50 flex justify-between items-center bg-slate-800/50">
                  <h3 className="font-semibold text-white">Annotation Results</h3>
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
                        <th className="p-4 font-medium hidden sm:table-cell">Top Matches</th>
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
                                  style={{ width: `${cell.confidence_score * 100}%` }}
                                />
                              </div>
                              <span className="text-xs">
                                {(cell.confidence_score * 100).toFixed(0)}%
                              </span>
                            </div>
                          </td>
                          <td className="p-4 text-xs text-slate-500 hidden sm:table-cell">
                            {cell.top_matches.slice(0, 2).map((m) => m.annotation).join(', ')}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </div>
          )}
        </div>
      </div>

      {!showChat && (
        <button
          type="button"
          onClick={() => setShowChat(true)}
          className="fixed bottom-8 right-8 z-50 bg-gradient-to-r from-indigo-600 to-cyan-600 hover:from-indigo-500 hover:to-cyan-500 text-white p-4 rounded-full shadow-lg shadow-indigo-500/30 transition-all duration-300 hover:scale-110 flex items-center justify-center group"
        >
          <Bot size={28} className="group-hover:rotate-12 transition-transform" />
        </button>
      )}

      <div
        className={`hidden lg:block fixed top-20 right-6 bottom-6 w-[600px] z-40 transition-all duration-500 ease-out ${
          showChat ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-6 pointer-events-none'
        }`}
        aria-hidden={!showChat}
      >
        <Card className="h-full flex flex-col bg-slate-900/95 border-slate-700/70 shadow-2xl">
          <ChatPage embedded onClose={() => setShowChat(false)} />
        </Card>
      </div>
    </div>
  );
};

export default PortalPage;

