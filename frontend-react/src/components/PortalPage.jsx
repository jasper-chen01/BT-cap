import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertCircle,
  BarChart3,
  Bot,
  CheckCircle,
  Cpu,
  Download,
  FileText,
  Filter,
  Layers,
  Palette,
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
  const [activeTab, setActiveTab] = useState('annotation');
  const [vizResults, setVizResults] = useState(null);
  const [isVisualizing, setIsVisualizing] = useState(false);
  const [vizStatus, setVizStatus] = useState(null);
  const [colorMode, setColorMode] = useState('cluster');
  const [selectedCellTypes, setSelectedCellTypes] = useState([]);
  const [minScore, setMinScore] = useState(0);
  const [deTopN, setDeTopN] = useState(15);
  const [clusterResolution, setClusterResolution] = useState(1.0);
  const [supptableUrl, setSupptableUrl] = useState('');
  const [deGroupby, setDeGroupby] = useState('cluster');
  const [deGroup, setDeGroup] = useState('');

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

  const showVizStatus = (message, type) => {
    setVizStatus({ message, type });
    if (type === 'success') {
      setTimeout(() => setVizStatus(null), 6000);
    }
  };

  useEffect(() => {
    if (!vizResults) return;
    if (vizResults.cell_types?.length) {
      setColorMode((prev) => (prev === 'cluster' ? 'cell_type' : prev));
    }
    setSelectedCellTypes([]);
    setMinScore(0);
    setDeGroupby('cluster');
  }, [vizResults]);

  const filteredUmapPoints = useMemo(() => {
    if (!vizResults?.umap_points) return [];
    const hasCellTypes = vizResults.cell_types?.length;
    return vizResults.umap_points.filter((point) => {
      if (hasCellTypes) {
        if (selectedCellTypes.length && !selectedCellTypes.includes(point.cell_type)) {
          return false;
        }
        if (minScore > 0) {
          if (typeof point.score !== 'number' || point.score < minScore) {
            return false;
          }
        }
      }
      return true;
    });
  }, [vizResults, selectedCellTypes, minScore]);

  const MAX_UMAP_POINTS = 40000;
  const sampledUmapPoints = useMemo(() => {
    if (filteredUmapPoints.length <= MAX_UMAP_POINTS) {
      return filteredUmapPoints;
    }
    const step = Math.ceil(filteredUmapPoints.length / MAX_UMAP_POINTS);
    return filteredUmapPoints.filter((_, index) => index % step === 0);
  }, [filteredUmapPoints]);

  const colorMap = useMemo(() => {
    if (!vizResults?.umap_points) return {};
    const palette = [
      '#38bdf8',
      '#f97316',
      '#a855f7',
      '#22c55e',
      '#eab308',
      '#f43f5e',
      '#14b8a6',
      '#6366f1',
      '#f59e0b',
      '#06b6d4',
      '#e879f9',
      '#84cc16',
    ];
    const labels = [];
    vizResults.umap_points.forEach((point) => {
      const label = colorMode === 'cell_type' ? point.cell_type : point.cluster;
      if (label && !labels.includes(label)) {
        labels.push(label);
      }
    });
    return labels.reduce((acc, label, index) => {
      acc[label] = palette[index % palette.length];
      return acc;
    }, {});
  }, [vizResults, colorMode]);

  const activeDeGroups = useMemo(() => {
    if (!vizResults) return [];
    return deGroupby === 'cell_type'
      ? vizResults.de_by_cell_type || []
      : vizResults.de_by_cluster || [];
  }, [vizResults, deGroupby]);

  const umapBounds = useMemo(() => {
    if (!sampledUmapPoints.length) return null;
    const xs = sampledUmapPoints.map((point) => point.x);
    const ys = sampledUmapPoints.map((point) => point.y);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys),
    };
  }, [sampledUmapPoints]);

  const umapSize = { width: 1000, height: 700 };

  useEffect(() => {
    if (!activeDeGroups.length) return;
    if (!activeDeGroups.find((group) => group.group === deGroup)) {
      setDeGroup(activeDeGroups[0].group);
    }
  }, [activeDeGroups, deGroup]);

  const activeDeGroup = useMemo(
    () => activeDeGroups.find((group) => group.group === deGroup),
    [activeDeGroups, deGroup]
  );

  const toggleCellType = (cellType) => {
    setSelectedCellTypes((prev) => {
      if (prev.includes(cellType)) {
        return prev.filter((value) => value !== cellType);
      }
      return [...prev, cellType];
    });
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

  const visualizeData = async () => {
    if (!file) {
      showVizStatus('Please select a .h5ad file first.', 'error');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('de_top_n', deTopN);
    formData.append('cluster_resolution', clusterResolution);
    if (supptableUrl.trim()) {
      formData.append('supptable_url', supptableUrl.trim());
    }

    setIsVisualizing(true);
    showVizStatus('Running Scanpy workflow (normalize, cluster, UMAP)...', 'info');

    try {
      const response = await fetch(`${API_URL}/visualize`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Visualization failed');
      }

      const data = await response.json();
      setVizResults(data);
      showVizStatus(`Visualization ready: ${data.total_cells} cells processed.`, 'success');
    } catch (error) {
      showVizStatus(`Error: ${error.message}`, 'error');
    } finally {
      setIsVisualizing(false);
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
          <p className="text-slate-400">Manage your datasets and run annotations or visualization</p>
        </div>
        <div className="flex gap-2 bg-slate-900/60 border border-slate-700/70 p-1 rounded-full">
          <button
            type="button"
            onClick={() => setActiveTab('annotation')}
            className={`px-4 py-1.5 rounded-full text-sm transition ${
              activeTab === 'annotation'
                ? 'bg-indigo-500 text-white'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            Annotation
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('visualization')}
            className={`px-4 py-1.5 rounded-full text-sm transition ${
              activeTab === 'visualization'
                ? 'bg-cyan-500 text-slate-900'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            Visualization
          </button>
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

          {activeTab === 'annotation' ? (
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
          ) : (
            <Card>
              <div className="flex items-center gap-2 mb-6">
                <Settings className="text-slate-400" size={20} />
                <h3 className="text-lg font-semibold text-white">Visualization Parameters</h3>
              </div>

              <div className="space-y-6">
                <div className="space-y-2">
                  <label className="text-sm text-slate-300" htmlFor="supptable-url">
                    Supptable URL (optional)
                  </label>
                  <input
                    id="supptable-url"
                    type="text"
                    value={supptableUrl}
                    onChange={(event) => setSupptableUrl(event.target.value)}
                    placeholder="Leave blank to use Firestore supptable1"
                    className="w-full rounded-lg bg-slate-900/60 border border-slate-700/70 px-3 py-2 text-sm text-slate-200"
                  />
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-300">Cluster Resolution</span>
                    <span className="text-cyan-300 font-mono">{clusterResolution.toFixed(1)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.2"
                    max="2"
                    step="0.1"
                    value={clusterResolution}
                    onChange={(event) => setClusterResolution(Number(event.target.value))}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                  />
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-300">DE Top Genes</span>
                    <span className="text-cyan-300 font-mono">{deTopN}</span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="50"
                    step="5"
                    value={deTopN}
                    onChange={(event) => setDeTopN(Number(event.target.value))}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                  />
                </div>

                <Button
                  onClick={visualizeData}
                  disabled={isVisualizing || !file}
                  className="w-full"
                  icon={isVisualizing ? Activity : Play}
                >
                  {isVisualizing ? 'Processing...' : 'Run Visualization'}
                </Button>
              </div>
            </Card>
          )}

          {activeTab === 'annotation' && status && (
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

          {activeTab === 'visualization' && vizStatus && (
            <div
              className={`p-4 rounded-xl border flex items-start gap-3 ${
                vizStatus.type === 'error'
                  ? 'bg-red-500/10 border-red-500/20 text-red-200'
                  : vizStatus.type === 'success'
                  ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-200'
                  : 'bg-blue-500/10 border-blue-500/20 text-blue-200'
              }`}
            >
              {vizStatus.type === 'error' ? (
                <AlertCircle size={20} />
              ) : vizStatus.type === 'success' ? (
                <CheckCircle size={20} />
              ) : (
                <Activity size={20} />
              )}
              <p className="text-sm">{vizStatus.message}</p>
            </div>
          )}
        </div>

        <div className="lg:col-span-2">
          {activeTab === 'annotation' ? (
            !results ? (
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
            )
          ) : !vizResults ? (
            <div className="h-full min-h-[400px] flex flex-col items-center justify-center border-2 border-dashed border-slate-700 rounded-2xl bg-slate-800/20 text-slate-500">
              <Cpu size={48} className="mb-4 opacity-50" />
              <p className="text-lg">Visualization will appear here</p>
              <p className="text-sm opacity-60">Upload a file and run visualization to begin</p>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <Card className="text-center p-4">
                  <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                    Total Cells
                  </p>
                  <p className="text-3xl font-bold text-white">
                    {vizResults.total_cells.toLocaleString()}
                  </p>
                </Card>
                <Card className="text-center p-4">
                  <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                    Clusters
                  </p>
                  <p className="text-3xl font-bold text-cyan-300">
                    {vizResults.cluster_labels.length}
                  </p>
                </Card>
                <Card className="text-center p-4">
                  <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                    Cell Types
                  </p>
                  <p className="text-3xl font-bold text-emerald-400">
                    {vizResults.cell_types?.length || 0}
                  </p>
                </Card>
              </div>

              <Card className="overflow-hidden p-0">
                <div className="p-4 border-b border-slate-700/50 flex flex-wrap justify-between items-center gap-4 bg-slate-800/50">
                  <div className="flex items-center gap-2">
                    <BarChart3 size={18} className="text-slate-400" />
                    <h3 className="font-semibold text-white">UMAP Projection</h3>
                  </div>
                  <div className="flex flex-wrap items-center gap-3 text-sm text-slate-300">
                    <label className="flex items-center gap-2">
                      <Palette size={16} className="text-slate-400" />
                      <span>Color by</span>
                    </label>
                    <select
                      value={colorMode}
                      onChange={(event) => setColorMode(event.target.value)}
                      className="bg-slate-900/70 border border-slate-700/70 rounded-lg px-2 py-1 text-sm text-slate-200"
                    >
                      <option value="cluster">Cluster</option>
                      {vizResults.cell_types?.length ? (
                        <option value="cell_type">Cell Type</option>
                      ) : null}
                    </select>
                    <span className="text-xs text-slate-400">
                      Showing {sampledUmapPoints.length.toLocaleString()} of{' '}
                      {filteredUmapPoints.length.toLocaleString()} filtered /{' '}
                      {vizResults.total_cells.toLocaleString()}
                    </span>
                  </div>
                </div>

                <div className="p-4 bg-slate-900/40">
                  {umapBounds ? (
                    <svg
                      viewBox={`0 0 ${umapSize.width} ${umapSize.height}`}
                      className="w-full h-[500px] bg-slate-950 rounded-xl border border-slate-800/80"
                    >
                      {sampledUmapPoints.map((point) => {
                        const { minX, maxX, minY, maxY } = umapBounds;
                        const xScale = (point.x - minX) / (maxX - minX || 1);
                        const yScale = (point.y - minY) / (maxY - minY || 1);
                        const x = xScale * umapSize.width;
                        const y = umapSize.height - yScale * umapSize.height;
                        const label = colorMode === 'cell_type' ? point.cell_type : point.cluster;
                        const color = label ? colorMap[label] || '#94a3b8' : '#94a3b8';
                        return (
                          <circle
                            key={`${point.cell_id}-${point.cluster}`}
                            cx={x}
                            cy={y}
                            r={2}
                            fill={color}
                            fillOpacity={0.8}
                          />
                        );
                      })}
                    </svg>
                  ) : (
                    <div className="h-[400px] flex items-center justify-center text-slate-500">
                      No UMAP points available
                    </div>
                  )}
                  {filteredUmapPoints.length > MAX_UMAP_POINTS && (
                    <p className="mt-3 text-xs text-slate-500">
                      Rendering {sampledUmapPoints.length.toLocaleString()} sampled points for
                      performance. Use filters to narrow the view.
                    </p>
                  )}
                </div>

                <div className="p-4 border-t border-slate-800/70 flex flex-wrap gap-3 text-xs text-slate-300">
                  {Object.entries(colorMap)
                    .slice(0, 10)
                    .map(([label, color]) => (
                      <div key={label} className="flex items-center gap-2">
                        <span className="inline-block w-3 h-3 rounded-full" style={{ background: color }} />
                        <span>{label}</span>
                      </div>
                    ))}
                  {Object.keys(colorMap).length > 10 && (
                    <span className="text-slate-500">
                      +{Object.keys(colorMap).length - 10} more
                    </span>
                  )}
                </div>
              </Card>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <div className="flex items-center gap-2 mb-4">
                    <Filter size={18} className="text-slate-400" />
                    <h3 className="font-semibold text-white">Cell Type Filters</h3>
                  </div>
                  {vizResults.cell_types?.length ? (
                    <>
                      <div className="flex flex-wrap gap-2 mb-3">
                        <Button
                          variant="secondary"
                          className="py-1 px-3 text-xs"
                          onClick={() => setSelectedCellTypes([])}
                        >
                          Clear Selection
                        </Button>
                        <span className="text-xs text-slate-500">
                          {selectedCellTypes.length
                            ? `${selectedCellTypes.length} selected`
                            : 'All cell types'}
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 max-h-56 overflow-y-auto pr-1">
                        {vizResults.cell_types.map((cellType) => (
                          <label
                            key={cellType.name}
                            className="flex items-center gap-2 text-sm text-slate-300"
                          >
                            <input
                              type="checkbox"
                              checked={selectedCellTypes.includes(cellType.name)}
                              onChange={() => toggleCellType(cellType.name)}
                              className="accent-cyan-500"
                            />
                            <span className="flex-1 truncate">{cellType.name}</span>
                            <span className="text-xs text-slate-500">{cellType.count}</span>
                          </label>
                        ))}
                      </div>
                      <div className="mt-5 space-y-3">
                        <div className="flex justify-between text-sm">
                          <span className="text-slate-300">Min Score</span>
                          <span className="text-cyan-300 font-mono">{minScore.toFixed(2)}</span>
                        </div>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={minScore}
                          onChange={(event) => setMinScore(Number(event.target.value))}
                          className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                        />
                      </div>
                    </>
                  ) : (
                    <p className="text-sm text-slate-500">
                      No cell types found in the supplemental table.
                    </p>
                  )}
                </Card>

                <Card className="overflow-hidden p-0">
                  <div className="p-4 border-b border-slate-700/60 bg-slate-800/50 flex flex-wrap items-center gap-3">
                    <div className="flex items-center gap-2">
                      <Layers size={18} className="text-slate-400" />
                      <h3 className="font-semibold text-white">Differential Expression</h3>
                    </div>
                    <select
                      value={deGroupby}
                      onChange={(event) => setDeGroupby(event.target.value)}
                      className="bg-slate-900/70 border border-slate-700/70 rounded-lg px-2 py-1 text-sm text-slate-200"
                    >
                      <option value="cluster">Cluster</option>
                      {vizResults.de_by_cell_type?.length ? (
                        <option value="cell_type">Cell Type</option>
                      ) : null}
                    </select>
                    {activeDeGroups.length ? (
                      <select
                        value={deGroup}
                        onChange={(event) => setDeGroup(event.target.value)}
                        className="bg-slate-900/70 border border-slate-700/70 rounded-lg px-2 py-1 text-sm text-slate-200"
                      >
                        {activeDeGroups.map((group) => (
                          <option key={group.group} value={group.group}>
                            {group.group}
                          </option>
                        ))}
                      </select>
                    ) : null}
                  </div>
                  <div className="p-4">
                    {activeDeGroup ? (
                      <div className="overflow-x-auto max-h-[360px]">
                        <table className="w-full text-left text-sm text-slate-300">
                          <thead className="text-slate-400 uppercase text-xs">
                            <tr>
                              <th className="pb-2 pr-4">Gene</th>
                              <th className="pb-2 pr-4">Score</th>
                              <th className="pb-2">LogFC</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-800/70">
                            {activeDeGroup.genes.map((gene, index) => (
                              <tr key={`${activeDeGroup.group}-${gene}`}>
                                <td className="py-2 pr-4 font-mono text-xs text-slate-200">
                                  {gene}
                                </td>
                                <td className="py-2 pr-4 text-slate-400 text-xs">
                                  {activeDeGroup.scores?.[index]?.toFixed(3) ?? '—'}
                                </td>
                                <td className="py-2 text-slate-400 text-xs">
                                  {activeDeGroup.logfoldchanges?.[index]?.toFixed(3) ?? '—'}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <p className="text-sm text-slate-500">
                        Differential expression results are unavailable for this selection.
                      </p>
                    )}
                  </div>
                </Card>
              </div>
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

