import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as XLSX from 'xlsx';
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
  Radio,
  Settings,
  Upload,
} from 'lucide-react';
import Button from './ui/Button';
import Card from './ui/Card';
import ChatPage from './ChatPage';

const API_URL = 'http://localhost:8000/api';

const EPHYS_FILE_EXTENSIONS = [
  '.csv',
  '.abf',
  '.nex',
  '.mat',
  '.nwb',
  '.edf',
  '.h5',
];

const PortalPage = () => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [topK, setTopK] = useState(10);
  const [threshold, setThreshold] = useState(0.7);
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnnotating, setIsAnnotating] = useState(false);
  const [isGeneSetScoring, setIsGeneSetScoring] = useState(false);
  const fileInputRef = useRef(null);
  const [showChat, setShowChat] = useState(false);
  const [activeTab, setActiveTab] = useState('annotation');
  const [annotationView, setAnnotationView] = useState('predictions');
  const [embeddingMatches, setEmbeddingMatches] = useState(null);
  const [embeddingMatchesStatus, setEmbeddingMatchesStatus] = useState(null);
  const [embeddingMatchesJob, setEmbeddingMatchesJob] = useState(null);
  const [vizResults, setVizResults] = useState(null);
  const [analysisSummaryPath, setAnalysisSummaryPath] = useState('');
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
  const [deFilterType, setDeFilterType] = useState('all');
  const [deSortBy, setDeSortBy] = useState('score');
  const [deSortDir, setDeSortDir] = useState('desc');
  const [deGeneSearch, setDeGeneSearch] = useState('');

  const [ephysFile, setEphysFile] = useState(null);
  const [ephysFileName, setEphysFileName] = useState('');
  const [ephysDragging, setEphysDragging] = useState(false);
  const [ephysStatus, setEphysStatus] = useState(null);
  const [ephysSamplingRateKhz, setEphysSamplingRateKhz] = useState(20);
  const [ephysFilterLowHz, setEphysFilterLowHz] = useState(300);
  const [ephysFilterHighHz, setEphysFilterHighHz] = useState(3000);
  const [ephysSnrThreshold, setEphysSnrThreshold] = useState(4);
  const [ephysShowPreview, setEphysShowPreview] = useState(false);
  const ephysInputRef = useRef(null);

  const [prepsH5adFile, setPrepsH5adFile] = useState(null);
  const [prepsH5adFileName, setPrepsH5adFileName] = useState('');
  const [prepsSpecies, setPrepsSpecies] = useState('human');
  const [prepsGpu, setPrepsGpu] = useState('0');
  const [prepsRefSubstring, setPrepsRefSubstring] = useState('');
  const [prepsJobId, setPrepsJobId] = useState(null);
  const [prepsJobPayload, setPrepsJobPayload] = useState(null);
  const [isPrepsRunning, setIsPrepsRunning] = useState(false);
  const [ephysVizResults, setEphysVizResults] = useState(null);
  const [prepsConfigured, setPrepsConfigured] = useState(null);
  const prepsH5adInputRef = useRef(null);
  const prepsEphysColorDefaultAppliedRef = useRef(false);

  const sourceViz =
    activeTab === 'electrophysiology' ? ephysVizResults : vizResults;

  const showMainVizPanel =
    (activeTab === 'visualization' && vizResults) ||
    (activeTab === 'electrophysiology' && ephysVizResults);

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

  useEffect(() => {
    if (activeTab !== 'electrophysiology') return;
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${API_URL}/preps/config`);
        const data = await r.json();
        if (!cancelled) setPrepsConfigured(data);
      } catch {
        if (!cancelled) setPrepsConfigured({ preps_available: false });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeTab]);

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

  const showEmbeddingStatus = (message, type = 'info') => {
    setEmbeddingMatchesStatus({ message, type });
  };

  const showVizStatus = (message, type) => {
    setVizStatus({ message, type });
    if (type === 'success') {
      setTimeout(() => setVizStatus(null), 6000);
    }
  };

  const showEphysStatus = (message, type) => {
    setEphysStatus({ message, type });
    if (type === 'success') {
      setTimeout(() => setEphysStatus(null), 6000);
    }
  };

  useEffect(() => {
    if (!prepsJobId || !isPrepsRunning) return undefined;
    let cancelled = false;

    const poll = async () => {
      try {
        const r = await fetch(`${API_URL}/preps/jobs/${prepsJobId}?include_log_tail=40`);
        if (!r.ok) return;
        const data = await r.json();
        if (cancelled) return;
        setPrepsJobPayload(data);
        if (data.status === 'succeeded' && data.visualization) {
          setEphysVizResults(data.visualization);
          setAnalysisSummaryPath(
            data.visualization?.metadata?.analysis_summary_path || '',
          );
          setIsPrepsRunning(false);
          const em = data.visualization?.metadata || {};
          let doneMsg =
            'PREPS finished. UMAP uses Scanpy; color by cell type to see Geneformer/PREPS labels.';
          if (em.ephys_patchseq_ran && em.ephys_patchseq_output_dir) {
            doneMsg += ` Predicted ephys tables: ${em.ephys_patchseq_output_dir}`;
          } else if (em.ephys_patchseq_note) {
            doneMsg += ` Ephys step: ${em.ephys_patchseq_note}`;
          }
          showEphysStatus(doneMsg, 'success');
        } else if (data.status === 'failed') {
          setIsPrepsRunning(false);
          showEphysStatus(data.error_message || 'PREPS job failed.', 'error');
        }
      } catch {
        /* keep polling */
      }
    };

    poll();
    const id = setInterval(poll, 3000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [prepsJobId, isPrepsRunning]);

  useEffect(() => {
    if (
      activeTab !== 'electrophysiology' ||
      prepsEphysColorDefaultAppliedRef.current ||
      !ephysVizResults?.metadata?.available_programs
    ) {
      return;
    }
    const programs = ephysVizResults.metadata.available_programs;
    const firstEphys = programs.find((p) => String(p).startsWith('Ephys ·'));
    if (firstEphys) {
      setColorMode(firstEphys);
      prepsEphysColorDefaultAppliedRef.current = true;
    }
  }, [activeTab, ephysVizResults]);

  useEffect(() => {
    if (!sourceViz) return;
    if (sourceViz.cell_types?.length) {
      setColorMode((prev) => (prev === 'cluster' ? 'cell_type' : prev));
    }
    setSelectedCellTypes([]);
    setMinScore(0);
    setDeGroupby('cluster');
  }, [sourceViz]);

  useEffect(() => {
    if (!results) {
      setAnnotationView('predictions');
      setEmbeddingMatches(null);
      setEmbeddingMatchesJob(null);
      setEmbeddingMatchesStatus(null);
      return;
    }
    setEmbeddingMatches(null);
    setEmbeddingMatchesJob(null);
    setEmbeddingMatchesStatus(null);
    setAnnotationView('predictions');
  }, [results]);

  const fetchEmbeddingMatches = async () => {
    const jobId = results?.metadata?.embedding_pipeline_job?.job_id;
    if (!jobId) {
      showEmbeddingStatus('Embedding pipeline has not started yet.', 'error');
      return;
    }

    showEmbeddingStatus('Checking embedding pipeline status...', 'info');
    try {
      const jobResponse = await fetch(
        `${API_URL}/embeddings/pipeline/jobs/${jobId}`
      );
      if (!jobResponse.ok) {
        const error = await jobResponse.json();
        throw new Error(error.detail || 'Failed to load pipeline status');
      }
      const job = await jobResponse.json();
      setEmbeddingMatchesJob(job);

      if (job.status !== 'succeeded') {
        showEmbeddingStatus(
          `Embedding pipeline ${job.status}. Check back once it finishes.`,
          job.status === 'failed' ? 'error' : 'info'
        );
        return;
      }

      const matchesResponse = await fetch(
        `${API_URL}/embeddings/pipeline/jobs/${jobId}/matches?kind=annotated`
      );
      if (!matchesResponse.ok) {
        const rawResponse = await fetch(
          `${API_URL}/embeddings/pipeline/jobs/${jobId}/matches?kind=raw`
        );
        if (!rawResponse.ok) {
          const error = await rawResponse.json();
          throw new Error(error.detail || 'Failed to load embedding matches');
        }
        const rawData = await rawResponse.json();
        setEmbeddingMatches(rawData);
        showEmbeddingStatus('Loaded raw embedding matches.', 'success');
        return;
      }

      const data = await matchesResponse.json();
      setEmbeddingMatches(data);
      showEmbeddingStatus('Embedding matches loaded.', 'success');
    } catch (error) {
      showEmbeddingStatus(`Error: ${error.message}`, 'error');
    }
  };

  const filteredUmapPoints = useMemo(() => {
    if (!sourceViz?.umap_points) return [];

    const hasCellTypes = sourceViz.cell_types?.length;
    const isProgramMode =
      colorMode !== 'cluster' && colorMode !== 'cell_type';

    return sourceViz.umap_points.filter((point) => {
      if (hasCellTypes && selectedCellTypes.length) {
        if (!selectedCellTypes.includes(point.cell_type)) {
          return false;
        }
      }

      if (isProgramMode && minScore > 0) {
        const score = point.program_scores?.[colorMode];
        if (typeof score !== 'number' || score < minScore) {
          return false;
        }
      }

      return true;
    });
  }, [sourceViz, selectedCellTypes, minScore, colorMode]);

  const MAX_UMAP_POINTS = 40000;
  const sampledUmapPoints = useMemo(() => {
    if (filteredUmapPoints.length <= MAX_UMAP_POINTS) {
      return filteredUmapPoints;
    }
    const step = Math.ceil(filteredUmapPoints.length / MAX_UMAP_POINTS);
    return filteredUmapPoints.filter((_, index) => index % step === 0);
  }, [filteredUmapPoints]);

  const colorMap = useMemo(() => {
    if (!sourceViz?.umap_points) return {};

    const isProgramMode =
      colorMode !== 'cluster' && colorMode !== 'cell_type';
    if (isProgramMode) return {};

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
    sourceViz.umap_points.forEach((point) => {
      const label = colorMode === 'cell_type' ? point.cell_type : point.cluster;
      if (label && !labels.includes(label)) {
        labels.push(label);
      }
    });

    return labels.reduce((acc, label, index) => {
      acc[label] = palette[index % palette.length];
      return acc;
    }, {});
  }, [sourceViz, colorMode]);

  const activeDeGroups = useMemo(() => {
    if (!sourceViz) return [];
    return deGroupby === 'cell_type'
      ? sourceViz.de_by_cell_type || []
      : sourceViz.de_by_cluster || [];
  }, [sourceViz, deGroupby]);

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

  useEffect(() => {
    if (annotationView === 'embeddings') {
      fetchEmbeddingMatches();
    }
  }, [annotationView]);

  const availablePrograms = useMemo(
    () => sourceViz?.metadata?.available_programs || [],
    [sourceViz]
  );

  const programScoreRange = useMemo(() => {
    if (!sourceViz?.umap_points?.length) return null;

    const isProgramMode =
      colorMode !== 'cluster' && colorMode !== 'cell_type';
    if (!isProgramMode) return null;

    const scores = sourceViz.umap_points
      .map((p) =>
        typeof p.program_scores?.[colorMode] === 'number'
          ? p.program_scores[colorMode]
          : null
      )
      .filter((v) => v !== null)
      .sort((a, b) => a - b);

    if (!scores.length) return null;

    const getPercentile = (arr, q) => {
      if (!arr.length) return null;
      const pos = (arr.length - 1) * q;
      const base = Math.floor(pos);
      const rest = pos - base;
      if (arr[base + 1] !== undefined) {
        return arr[base] + rest * (arr[base + 1] - arr[base]);
      }
      return arr[base];
    };

    return {
      rawMin: scores[0],
      rawMax: scores[scores.length - 1],
      displayMin: getPercentile(scores, 0.05),
      displayMax: getPercentile(scores, 0.95),
    };
  }, [sourceViz, colorMode]);

  const overlapReport = sourceViz?.metadata?.de_overlaps;

  const formatDePadj = (p) => {
    if (p == null || Number.isNaN(p)) return '—';
    if (p === 0) return '0';
    if (p < 1e-6) return p.toExponential(2);
    if (p < 0.001) return '<0.001';
    return p.toFixed(4);
  };

  const deRows = useMemo(() => {
    if (!activeDeGroup) return [];
    const rows = activeDeGroup.genes.map((gene, index) => {
      const geneAnnotation = activeDeGroup.gene_annotations?.[index] ?? null;
      const hasLigand = Boolean(geneAnnotation?.is_ligand);
      const hasReceptor = Boolean(geneAnnotation?.is_receptor);
      const drugTargets = geneAnnotation?.drug_targets;
      const drugCount = Array.isArray(drugTargets) ? drugTargets.length : 0;
      const score = activeDeGroup.scores?.[index];
      const logfc = activeDeGroup.logfoldchanges?.[index];
      const padj = activeDeGroup.pvals_adj?.[index];
      return {
        gene,
        index,
        geneAnnotation,
        hasLigand,
        hasReceptor,
        drugTargets,
        drugCount,
        score: typeof score === 'number' && !Number.isNaN(score) ? score : null,
        logfc: typeof logfc === 'number' && !Number.isNaN(logfc) ? logfc : null,
        padj: typeof padj === 'number' && !Number.isNaN(padj) ? padj : null,
      };
    });

    const search = deGeneSearch.trim().toLowerCase();
    const filtered = rows.filter((row) => {
      if (deFilterType === 'ligand' && !row.hasLigand) return false;
      if (deFilterType === 'receptor' && !row.hasReceptor) return false;
      if (deFilterType === 'druggable' && row.drugCount === 0) return false;
      if (search && !row.gene.toLowerCase().includes(search)) return false;
      return true;
    });

    const sortMultiplier = deSortDir === 'asc' ? 1 : -1;
    const scoreValue = (value) =>
      typeof value === 'number' && !Number.isNaN(value) ? value : -Infinity;
    const pvalSort = (value) =>
      typeof value === 'number' && !Number.isNaN(value) ? value : Infinity;
    return filtered.sort((a, b) => {
      if (deSortBy === 'gene') {
        return sortMultiplier * a.gene.localeCompare(b.gene);
      }
      if (deSortBy === 'logfc') {
        return sortMultiplier * (scoreValue(a.logfc) - scoreValue(b.logfc));
      }
      if (deSortBy === 'padj') {
        return sortMultiplier * (pvalSort(a.padj) - pvalSort(b.padj));
      }
      if (deSortBy === 'drug_count') {
        return sortMultiplier * (a.drugCount - b.drugCount);
      }
      return sortMultiplier * (scoreValue(a.score) - scoreValue(b.score));
    });
  }, [
    activeDeGroup,
    deFilterType,
    deSortBy,
    deSortDir,
    deGeneSearch,
  ]);

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

  const processEphysFile = (selectedFile) => {
    if (!selectedFile) return;
    const lower = selectedFile.name.toLowerCase();
    const ok = EPHYS_FILE_EXTENSIONS.some((ext) => lower.endsWith(ext));
    if (ok) {
      setEphysFile(selectedFile);
      setEphysFileName(selectedFile.name);
      setEphysShowPreview(false);
      showEphysStatus('Recording file selected.', 'success');
    } else {
      showEphysStatus(
        `Unsupported format. Use: ${EPHYS_FILE_EXTENSIONS.join(', ')}`,
        'error',
      );
    }
  };

  const handleEphysFileChange = (event) => {
    const selectedFile = event.target.files?.[0] || null;
    processEphysFile(selectedFile);
  };

  const processPrepsH5adFile = (selectedFile) => {
    if (!selectedFile) return;
    const lower = selectedFile.name.toLowerCase();
    if (lower.endsWith('.h5ad')) {
      setPrepsH5adFile(selectedFile);
      setPrepsH5adFileName(selectedFile.name);
      showEphysStatus('.h5ad selected for PREPS + UMAP.', 'success');
    } else {
      showEphysStatus('PREPS expects a .h5ad file.', 'error');
    }
  };

  const handlePrepsH5adChange = (event) => {
    processPrepsH5adFile(event.target.files?.[0] || null);
  };

  const runPrepsPipeline = async () => {
    if (!prepsH5adFile) {
      showEphysStatus('Select a .h5ad file for PREPS first.', 'error');
      return;
    }
    if (prepsConfigured && prepsConfigured.preps_available === false) {
      showEphysStatus(
        'Set PREPS_PYTHON in backend .env to your conda preps interpreter (see preps/HOWTO_PREPS.md).',
        'error',
      );
      return;
    }
    setIsPrepsRunning(true);
    setPrepsJobPayload(null);
    setEphysVizResults(null);
    setPrepsJobId(null);
    prepsEphysColorDefaultAppliedRef.current = false;
    showEphysStatus(
      'PREPS job queued (tokenize + annotate + patchseq_predict ephys). This can take a long time — polling every 3s.',
      'info',
    );
    try {
      const formData = new FormData();
      formData.append('file', prepsH5adFile);
      formData.append('species', prepsSpecies);
      formData.append('gpu', prepsGpu);
      formData.append('de_top_n', String(deTopN));
      formData.append('cluster_resolution', String(clusterResolution));
      if (prepsRefSubstring.trim()) {
        formData.append('reference_substring', prepsRefSubstring.trim());
      }
      const r = await fetch(`${API_URL}/preps/jobs`, { method: 'POST', body: formData });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) {
        const d = body.detail;
        let msg = r.statusText;
        if (typeof d === 'string') msg = d;
        else if (Array.isArray(d)) msg = d.map((x) => x?.msg || JSON.stringify(x)).join('; ');
        else if (d != null) msg = JSON.stringify(d);
        throw new Error(msg);
      }
      setPrepsJobId(body.job_id);
    } catch (e) {
      setIsPrepsRunning(false);
      showEphysStatus(e.message || 'Failed to start PREPS job.', 'error');
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    setEphysDragging(false);
    const droppedFile = event.dataTransfer.files?.[0] || null;
    if (activeTab === 'electrophysiology') {
      const name = droppedFile?.name?.toLowerCase() || '';
      if (name.endsWith('.h5ad')) {
        processPrepsH5adFile(droppedFile);
      } else {
        processEphysFile(droppedFile);
      }
    } else {
      processFile(droppedFile);
    }
  };

  const runEphysAnalysis = () => {
    if (!ephysFile) {
      showEphysStatus('Please select a recording file first.', 'error');
      return;
    }
    showEphysStatus(
      'Preview only: connect the electrophysiology API to run real analyses.',
      'info',
    );
    setEphysShowPreview(true);
  };

  const buildVisualizeFormData = () => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('de_top_n', deTopN);
    formData.append('cluster_resolution', clusterResolution);
    if (supptableUrl.trim()) {
      formData.append('supptable_url', supptableUrl.trim());
    }
    return formData;
  };

  const applyVisualizationResult = (data) => {
    setVizResults(data);
    setAnalysisSummaryPath(data?.metadata?.analysis_summary_path || '');
    const selected = data?.metadata?.selected_program || '';
    if (selected) {
      setColorMode(selected);
    } else if (data?.cell_types?.length) {
      setColorMode('cell_type');
    } else {
      setColorMode('cluster');
    }
  };

  const postVisualize = async (formData) => {
    const response = await fetch(`${API_URL}/visualize`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Visualization failed');
    }
    return response.json();
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
      const pipelineJobId = data?.metadata?.embedding_pipeline_job?.job_id;
      const pipelineStatus = data?.metadata?.embedding_pipeline_job?.status;
      let message = `Analysis complete: ${data.total_cells} cells annotated.`;
      if (pipelineJobId) {
        message += ` Pipeline job started: ${pipelineJobId}.`;
      } else if (pipelineStatus && pipelineStatus !== 'not_started') {
        message += ` Pipeline status: ${pipelineStatus}.`;
      }
      showStatus(message, 'success');
    } catch (error) {
      showStatus(`Error: ${error.message}`, 'error');
    } finally {
      setIsAnnotating(false);
    }
  };

  const runGeneSetScoring = async () => {
    if (!file) {
      showStatus('Please select a .h5ad file first.', 'error');
      return;
    }

    setIsGeneSetScoring(true);
    showStatus(
      'Running gene set scoring (Scanpy + supptable program scores; may take a while)...',
      'info',
    );

    try {
      const data = await postVisualize(buildVisualizeFormData());
      applyVisualizationResult(data);
      setActiveTab('visualization');
      showVizStatus(
        `Gene set scoring complete: ${data.total_cells} cells. Color the UMAP by program scores in Visualization.`,
        'success',
      );
    } catch (error) {
      showStatus(`Gene set scoring error: ${error.message}`, 'error');
    } finally {
      setIsGeneSetScoring(false);
    }
  };

  const visualizeData = async () => {
    if (!file) {
      showVizStatus('Please select a .h5ad file first.', 'error');
      return;
    }

    setIsVisualizing(true);
    showVizStatus('Running Scanpy workflow (normalize, cluster, UMAP)...', 'info');

    try {
      const data = await postVisualize(buildVisualizeFormData());
      applyVisualizationResult(data);
      showVizStatus(`Visualization ready: ${data.total_cells} cells processed.`, 'success');
    } catch (error) {
      showVizStatus(`Error: ${error.message}`, 'error');
    } finally {
      setIsVisualizing(false);
    }
  };

  const downloadProgramScores = () => {
    if (!sourceViz?.umap_points?.length) return;
    const rows = sourceViz.umap_points.map((point) => ({
      cell_id: point.cell_id,
      cluster: point.cluster,
      cell_type: point.cell_type ?? null,
      cell_type_score: typeof point.score === 'number' ? point.score : null,
      predicted_cell_type: point.predicted_cell_type ?? null,
      predicted_score: typeof point.predicted_score === 'number' ? point.predicted_score : null,
      umap_x: point.x,
      umap_y: point.y,
    }));
    const worksheet = XLSX.utils.json_to_sheet(rows);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'program_scores');
    const stamp = new Date().toISOString().split('T')[0];
    XLSX.writeFile(workbook, `program_scores_${stamp}.xlsx`);
  };

  const downloadResults = () => {
    if (!results) return;

    const headers = ['Cell ID', 'Predicted Annotation', 'Confidence Score (%)', 'Top Matches'];
    const rows = results.annotations.map((cell) => [
      cell.cell_id,
      cell.predicted_annotation,
      (cell.confidence_score * 100).toFixed(6),
      cell.top_matches
        .map((match) => `${match.annotation}:${(match.similarity * 100).toFixed(6)}%`)
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

  const downloadEmbeddingMatches = () => {
    if (!embeddingMatches?.rows?.length) return;
    const headers =
      embeddingMatches.columns || Object.keys(embeddingMatches.rows[0]);
    const rows = embeddingMatches.rows.map((row) =>
      headers.map((header) => {
        const value = row?.[header];
        return value === null || value === undefined ? '' : String(value);
      })
    );

    const csvContent = [
      headers.join(','),
      ...rows.map((row) => row.map((value) => `"${value}"`).join(',')),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `embedding_matches_${new Date().toISOString().split('T')[0]}.csv`;
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
          <p className="text-slate-400">
            Annotation, visualization, or electrophysiology — including PREPS + UMAP on the
            Electrophysiology tab when the backend PREPS env is configured
          </p>
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
          <button
            type="button"
            onClick={() => setActiveTab('electrophysiology')}
            className={`px-4 py-1.5 rounded-full text-sm transition ${
              activeTab === 'electrophysiology'
                ? 'bg-amber-500 text-slate-900'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            Electrophysiology
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mt-8">
        <div className="lg:col-span-1 space-y-6">
          <Card
            className={`relative overflow-hidden transition-all duration-300 group ${
              activeTab === 'electrophysiology'
                ? ephysDragging
                  ? 'border-amber-500 bg-amber-500/10'
                  : ''
                : isDragging
                  ? 'border-indigo-500 bg-indigo-500/10'
                  : ''
            }`}
          >
            <div
              onDragOver={(event) => {
                event.preventDefault();
                if (activeTab === 'electrophysiology') {
                  setEphysDragging(true);
                  setIsDragging(false);
                } else {
                  setIsDragging(true);
                  setEphysDragging(false);
                }
              }}
              onDragLeave={() => {
                setIsDragging(false);
                setEphysDragging(false);
              }}
              onDrop={handleDrop}
              className={`text-center p-8 border-2 border-dashed border-slate-600 rounded-xl transition-colors ${
                activeTab === 'electrophysiology'
                  ? 'hover:border-amber-500/50'
                  : 'hover:border-indigo-500/50'
              }`}
            >
              {activeTab === 'electrophysiology' ? (
                <>
                  <div className="mb-4 flex justify-center">
                    <div className="p-4 bg-slate-700/50 rounded-full group-hover:scale-110 transition-transform duration-300">
                      {ephysFile ? (
                        <Radio className="text-amber-400" size={32} />
                      ) : (
                        <Upload className="text-amber-400/90" size={32} />
                      )}
                    </div>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">
                    {ephysFile ? 'Recording selected' : 'Upload recording'}
                  </h3>
                  <p className="text-sm text-slate-400 mb-6">
                    {ephysFileName ||
                      `Traces: ${EPHYS_FILE_EXTENSIONS.join(', ')} — or drop a .h5ad for PREPS (same zone)`}
                  </p>
                  <div className="relative">
                    <input
                      type="file"
                      accept={EPHYS_FILE_EXTENSIONS.join(',')}
                      onChange={handleEphysFileChange}
                      ref={ephysInputRef}
                      className="hidden"
                    />
                    <Button
                      variant="secondary"
                      className="w-full border-amber-700/40 hover:border-amber-500/60"
                      type="button"
                      onClick={() => ephysInputRef.current?.click()}
                    >
                      Browse files
                    </Button>
                  </div>
                </>
              ) : (
                <>
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
                      accept=".h5ad,.h5"
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
                </>
              )}
            </div>
          </Card>

          {activeTab === 'electrophysiology' && (
            <Card>
              <div className="flex items-center gap-2 mb-4">
                <Layers className="text-amber-400" size={20} />
                <h3 className="text-lg font-semibold text-white">PREPS + UMAP</h3>
              </div>
              <p className="text-xs text-slate-500 mb-4">
                Runs <span className="font-mono text-slate-400">preps/generate_preds.py</span> then{' '}
                <span className="font-mono text-slate-400">preps/patchseq_predict.py</span> (ephys
                predictions) via your conda <span className="font-mono text-slate-400">preps</span>{' '}
                interpreter (<span className="font-mono text-slate-400">PREPS_PYTHON</span>). The
                server merges PREPS scores into the object, then runs the same Scanpy UMAP workflow
                as the Visualization tab. Set{' '}
                <span className="font-mono text-slate-400">PREPS_RUN_PATCHSEQ_PREDICT=0</span> in
                .env to skip the ephys step.
              </p>
              {prepsConfigured && prepsConfigured.preps_available === false ? (
                <p className="text-xs text-amber-100/90 mb-4 rounded-lg border border-amber-600/40 bg-amber-500/10 px-3 py-2">
                  PREPS is not available: set PREPS_PYTHON and verify PREPS_MODELS_ROOT /
                  PREPS_DICT_DIR. See <span className="font-mono">preps/HOWTO_PREPS.md</span>.
                </p>
              ) : null}
              <div className="space-y-3 mb-4">
                <p className="text-sm text-slate-300">Input .h5ad</p>
                <p className="text-xs text-slate-500 font-mono truncate">
                  {prepsH5adFileName || 'None selected'}
                </p>
                <input
                  ref={prepsH5adInputRef}
                  type="file"
                  accept=".h5ad,.h5"
                  className="hidden"
                  onChange={handlePrepsH5adChange}
                />
                <Button
                  type="button"
                  variant="secondary"
                  className="w-full border-amber-700/40"
                  onClick={() => prepsH5adInputRef.current?.click()}
                >
                  Choose .h5ad
                </Button>
              </div>
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div>
                  <label className="text-xs text-slate-400 block mb-1">Species</label>
                  <select
                    value={prepsSpecies}
                    onChange={(e) => setPrepsSpecies(e.target.value)}
                    className="w-full rounded-lg bg-slate-900/60 border border-slate-700/70 px-2 py-1.5 text-sm text-slate-200"
                  >
                    <option value="human">human</option>
                    <option value="mouse">mouse</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs text-slate-400 block mb-1">GPU id (-g)</label>
                  <input
                    value={prepsGpu}
                    onChange={(e) => setPrepsGpu(e.target.value)}
                    className="w-full rounded-lg bg-slate-900/60 border border-slate-700/70 px-2 py-1.5 text-sm text-slate-200 font-mono"
                  />
                </div>
              </div>
              <div className="mb-4">
                <label className="text-xs text-slate-400 block mb-1">
                  Scores CSV substring (optional)
                </label>
                <input
                  value={prepsRefSubstring}
                  onChange={(e) => setPrepsRefSubstring(e.target.value)}
                  placeholder="e.g. dirks_primary_gbm_combined"
                  className="w-full rounded-lg bg-slate-900/60 border border-slate-700/70 px-2 py-1.5 text-sm text-slate-200"
                />
              </div>
              <Button
                type="button"
                onClick={runPrepsPipeline}
                disabled={!prepsH5adFile || isPrepsRunning}
                className="w-full bg-amber-600 hover:bg-amber-500 text-slate-900"
                icon={isPrepsRunning ? Activity : Play}
              >
                {isPrepsRunning ? 'PREPS running...' : 'Run PREPS + UMAP'}
              </Button>
              {prepsJobPayload?.status ? (
                <p className="mt-3 text-xs text-slate-500">
                  Status: {prepsJobPayload.status}
                  {prepsJobPayload.test_name ? ` · ${prepsJobPayload.test_name}` : ''}
                </p>
              ) : null}
              {prepsJobPayload?.log_tail ? (
                <pre className="mt-2 max-h-32 overflow-auto rounded-lg bg-slate-950/80 p-2 text-[10px] text-slate-400 whitespace-pre-wrap">
                  {prepsJobPayload.log_tail}
                </pre>
              ) : null}
            </Card>
          )}

          {activeTab === 'annotation' && (
            <Card>
              <div className="flex items-center gap-2 mb-6">
                <Settings className="text-slate-400" size={20} />
                <h3 className="text-lg font-semibold text-white">Parameters</h3>
              </div>

              <div className="space-y-8">
                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-semibold text-white mb-1">
                      Reference-based annotation
                    </h4>
                    <p className="text-xs text-slate-500 mb-3">
                      Transformer similarity search against the reference atlas.
                    </p>
                  </div>
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
                    disabled={
                      isAnnotating ||
                      isGeneSetScoring ||
                      isVisualizing ||
                      !file
                    }
                    className="w-full"
                    icon={isAnnotating ? Activity : Play}
                  >
                    {isAnnotating ? 'Processing...' : 'Run Reference based Annotation'}
                  </Button>
                </div>

                <div className="border-t border-slate-700/80 pt-6 space-y-4">
                  <div>
                    <h4 className="text-sm font-semibold text-white mb-1">Gene set scoring</h4>
                    <p className="text-xs text-slate-500 mb-1">
                      LLM-derived program genes from the supptable (server{' '}
                      <span className="font-mono text-slate-400">data/SuppTable1.xlsx</span> when
                      present, or optional URL under Visualization).
                    </p>
                    <p className="text-xs text-slate-500">
                      Runs the same Scanpy + scoring pipeline as Visualization, then opens that tab
                      to explore program scores on the UMAP.
                    </p>
                  </div>

                  <Button
                    onClick={runGeneSetScoring}
                    disabled={
                      isGeneSetScoring ||
                      isAnnotating ||
                      isVisualizing ||
                      !file
                    }
                    className="w-full bg-violet-600 hover:bg-violet-500"
                    icon={isGeneSetScoring ? Activity : Play}
                  >
                    {isGeneSetScoring ? 'Processing...' : 'Run Gene Set Scoring'}
                  </Button>
                </div>
              </div>
            </Card>
          )}

          {activeTab === 'visualization' && (
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
                  disabled={
                    isVisualizing || isGeneSetScoring || isAnnotating || !file
                  }
                  className="w-full"
                  icon={isVisualizing ? Activity : Play}
                >
                  {isVisualizing ? 'Processing...' : 'Run Visualization'}
                </Button>
              </div>
            </Card>
          )}

          {activeTab === 'electrophysiology' && (
            <Card>
              <div className="flex items-center gap-2 mb-6">
                <Settings className="text-slate-400" size={20} />
                <h3 className="text-lg font-semibold text-white">Acquisition & detection</h3>
              </div>
              <p className="text-xs text-slate-500 mb-6">
                Frontend preview only — parameters will map to the electrophysiology service once
                the backend is available.
              </p>
              <div className="space-y-6">
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-300">Sampling rate (kHz)</span>
                    <span className="text-amber-300 font-mono">{ephysSamplingRateKhz}</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="50"
                    step="1"
                    value={ephysSamplingRateKhz}
                    onChange={(event) =>
                      setEphysSamplingRateKhz(Number(event.target.value))
                    }
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
                  />
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-300">Bandpass low (Hz)</span>
                    <span className="text-amber-300 font-mono">{ephysFilterLowHz}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="1000"
                    step="1"
                    value={ephysFilterLowHz}
                    onChange={(event) => setEphysFilterLowHz(Number(event.target.value))}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
                  />
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-300">Bandpass high (Hz)</span>
                    <span className="text-amber-300 font-mono">{ephysFilterHighHz}</span>
                  </div>
                  <input
                    type="range"
                    min="500"
                    max="10000"
                    step="100"
                    value={ephysFilterHighHz}
                    onChange={(event) => setEphysFilterHighHz(Number(event.target.value))}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
                  />
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-300">Spike detection (SNR threshold)</span>
                    <span className="text-amber-300 font-mono">{ephysSnrThreshold.toFixed(1)}</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="10"
                    step="0.5"
                    value={ephysSnrThreshold}
                    onChange={(event) => setEphysSnrThreshold(Number(event.target.value))}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
                  />
                </div>
                <Button
                  onClick={runEphysAnalysis}
                  disabled={!ephysFile}
                  className="w-full bg-amber-600 hover:bg-amber-500 text-slate-900"
                  icon={Play}
                >
                  Run analysis (preview)
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

          {activeTab === 'electrophysiology' && ephysStatus && (
            <div
              className={`p-4 rounded-xl border flex items-start gap-3 ${
                ephysStatus.type === 'error'
                  ? 'bg-red-500/10 border-red-500/20 text-red-200'
                  : ephysStatus.type === 'success'
                  ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-200'
                  : 'bg-amber-500/10 border-amber-500/25 text-amber-100'
              }`}
            >
              {ephysStatus.type === 'error' ? (
                <AlertCircle size={20} />
              ) : ephysStatus.type === 'success' ? (
                <CheckCircle size={20} />
              ) : (
                <Activity size={20} />
              )}
              <p className="text-sm">{ephysStatus.message}</p>
            </div>
          )}

        </div>

        <div className="lg:col-span-2">
          {activeTab === 'annotation' && (
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
                  <div className="p-4 border-b border-slate-700/50 flex flex-wrap justify-between items-center gap-4 bg-slate-800/50">
                    <div className="flex flex-wrap items-center gap-3">
                      <h3 className="font-semibold text-white">
                        {annotationView === 'predictions'
                          ? 'Annotation Results'
                          : 'Embedding Matches'}
                      </h3>
                      <div className="flex gap-2 bg-slate-900/60 border border-slate-700/70 p-1 rounded-full">
                        <button
                          type="button"
                          onClick={() => setAnnotationView('predictions')}
                          className={`px-3 py-1 rounded-full text-xs transition ${
                            annotationView === 'predictions'
                              ? 'bg-indigo-500 text-white'
                              : 'text-slate-300 hover:text-white'
                          }`}
                        >
                          Annotations
                        </button>
                        <button
                          type="button"
                          onClick={() => setAnnotationView('embeddings')}
                          className={`px-3 py-1 rounded-full text-xs transition ${
                            annotationView === 'embeddings'
                              ? 'bg-indigo-500 text-white'
                              : 'text-slate-300 hover:text-white'
                          }`}
                        >
                          Embeddings
                        </button>
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      {annotationView === 'predictions' ? (
                        <Button
                          variant="secondary"
                          onClick={downloadResults}
                          icon={Download}
                          className="py-1.5 px-4 text-sm"
                        >
                          Export CSV
                        </Button>
                      ) : (
                        <>
                          <Button
                            variant="secondary"
                            onClick={fetchEmbeddingMatches}
                            className="py-1.5 px-4 text-sm"
                          >
                            Refresh
                          </Button>
                          <Button
                            variant="secondary"
                            onClick={downloadEmbeddingMatches}
                            icon={Download}
                            className="py-1.5 px-4 text-sm"
                            disabled={!embeddingMatches?.rows?.length}
                          >
                            Export CSV
                          </Button>
                        </>
                      )}
                    </div>
                  </div>

                  {annotationView === 'predictions' ? (
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
                                    {(cell.confidence_score * 100).toFixed(6)}%
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
                  ) : (
                    <div className="p-4 space-y-4">
                      {embeddingMatchesStatus && (
                        <div
                          className={`p-3 rounded-xl border text-sm ${
                            embeddingMatchesStatus.type === 'error'
                              ? 'bg-red-500/10 border-red-500/20 text-red-200'
                              : embeddingMatchesStatus.type === 'success'
                              ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-200'
                              : 'bg-blue-500/10 border-blue-500/20 text-blue-200'
                          }`}
                        >
                          {embeddingMatchesStatus.message}
                        </div>
                      )}
                      {embeddingMatches?.rows?.length ? (
                        <div className="overflow-x-auto max-h-[600px] border border-slate-700/50 rounded-xl">
                          <table className="w-full text-left text-sm text-slate-300">
                            <thead className="bg-slate-900/60 text-slate-400 sticky top-0 z-10">
                              <tr>
                                {embeddingMatches.columns.map((col) => (
                                  <th key={col} className="p-3 font-medium">
                                    {col}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-700/50">
                              {embeddingMatches.rows.map((row, index) => (
                                <tr
                                  key={index}
                                  className="hover:bg-slate-700/30 transition-colors"
                                >
                                  {embeddingMatches.columns.map((col) => (
                                    <td key={`${col}-${index}`} className="p-3 text-xs">
                                      {row?.[col] ?? '—'}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <p className="text-sm text-slate-400">
                          Embedding matches will appear here once the pipeline finishes.
                        </p>
                      )}
                      {embeddingMatchesJob?.status && (
                        <p className="text-xs text-slate-500">
                          Pipeline status: {embeddingMatchesJob.status}
                        </p>
                      )}
                    </div>
                  )}
                </Card>
              </div>
            )
          )}

          {activeTab === 'visualization' && !vizResults && (
            <div className="h-full min-h-[400px] flex flex-col items-center justify-center border-2 border-dashed border-slate-700 rounded-2xl bg-slate-800/20 text-slate-500">
              <Cpu size={48} className="mb-4 opacity-50" />
              <p className="text-lg">Visualization will appear here</p>
              <p className="text-sm opacity-60">Upload a file and run visualization to begin</p>
            </div>
          )}

          {showMainVizPanel && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <Card className="text-center p-4">
                  <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                    Total Cells
                  </p>
                  <p className="text-3xl font-bold text-white">
                    {sourceViz.total_cells.toLocaleString()}
                  </p>
                </Card>
                <Card className="text-center p-4">
                  <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                    Clusters
                  </p>
                  <p className="text-3xl font-bold text-cyan-300">
                    {sourceViz.cluster_labels.length}
                  </p>
                </Card>
                <Card className="text-center p-4">
                  <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                    Cell Types
                  </p>
                  <p className="text-3xl font-bold text-emerald-400">
                    {sourceViz.cell_types?.length || 0}
                  </p>
                </Card>
              </div>

              <Card className="overflow-hidden p-0">
                <div className="p-4 border-b border-slate-700/50 flex flex-wrap justify-between items-center gap-4 bg-slate-800/50">
                  <div className="flex items-center gap-2">
                    <BarChart3 size={18} className="text-slate-400" />
                    <h3 className="font-semibold text-white">UMAP Projection</h3>
                    {activeTab === 'electrophysiology' && ephysVizResults ? (
                      <span className="text-xs font-normal text-amber-300/90 rounded-full border border-amber-600/40 px-2 py-0.5">
                        PREPS + Scanpy
                      </span>
                    ) : null}
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
                      {sourceViz.cell_types?.length ? (
                        <option value="cell_type">Cell Type</option>
                      ) : null}
                      {availablePrograms.map((program) => (
                        <option key={program} value={program}>
                          {program}
                        </option>
                      ))}
                    </select>
                    <span className="text-xs text-slate-400">
                      Showing {sampledUmapPoints.length.toLocaleString()} of{' '}
                      {filteredUmapPoints.length.toLocaleString()} filtered /{' '}
                      {sourceViz.total_cells.toLocaleString()}
                    </span>
                    <Button
                      variant="secondary"
                      onClick={downloadProgramScores}
                      icon={Download}
                      className="py-1.5 px-3 text-xs"
                    >
                      Export Scores
                    </Button>
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

                        const isProgramMode =
                          colorMode !== 'cluster' && colorMode !== 'cell_type';
                        const label =
                          colorMode === 'cell_type' ? point.cell_type : point.cluster;

                        let color = '#94a3b8';

                        if (isProgramMode) {
                          const s =
                            typeof point.program_scores?.[colorMode] === 'number'
                              ? point.program_scores[colorMode]
                              : null;

                          if (s !== null && programScoreRange) {
                            const t =
                              (s - programScoreRange.displayMin) /
                              (programScoreRange.displayMax - programScoreRange.displayMin || 1);
                            const clamped = Math.max(0, Math.min(1, t));
                            if (clamped < 0.33) {
                              const local = clamped / 0.33;
                              const r = Math.round(229 + local * (250 - 229));
                              const g = Math.round(231 + local * (204 - 231));
                              const b = Math.round(235 + local * (21 - 235));
                              color = `rgb(${r},${g},${b})`;
                            } else if (clamped < 0.66) {
                              const local = (clamped - 0.33) / 0.33;
                              const r = Math.round(250 + local * (249 - 250));
                              const g = Math.round(204 + local * (115 - 204));
                              const b = Math.round(21 + local * (22 - 21));
                              color = `rgb(${r},${g},${b})`;
                            } else {
                              const local = (clamped - 0.66) / 0.34;
                              const r = Math.round(249 + local * (220 - 249));
                              const g = Math.round(115 + local * (38 - 115));
                              const b = Math.round(22 + local * (38 - 22));
                              color = `rgb(${r},${g},${b})`;
                            }
                          } else {
                            color = '#cbd5e1';
                          }
                        } else {
                          color = label ? colorMap[label] || '#94a3b8' : '#94a3b8';
                        }

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

                {colorMode === 'cluster' || colorMode === 'cell_type' ? (
                  <div className="p-4 border-t border-slate-800/70 flex flex-wrap gap-3 text-xs text-slate-300">
                    {Object.entries(colorMap)
                      .slice(0, 10)
                      .map(([label, color]) => (
                        <div key={label} className="flex items-center gap-2">
                          <span
                            className="inline-block w-3 h-3 rounded-full"
                            style={{ background: color }}
                          />
                          <span>{label}</span>
                        </div>
                      ))}
                    {Object.keys(colorMap).length > 10 && (
                      <span className="text-slate-500">
                        +{Object.keys(colorMap).length - 10} more
                      </span>
                    )}
                  </div>
                ) : (
                  <div className="p-4 border-t border-slate-800/70 space-y-2 text-xs text-slate-400">
                    <div>
                      Continuous gene-program score for{' '}
                      <span className="text-slate-200">{colorMode}</span>
                    </div>

                    <div className="flex items-center gap-3">
                      <span className="text-slate-500">Low</span>
                      <div
                        className="h-3 w-48 rounded"
                        style={{
                          background:
                            'linear-gradient(to right, rgb(229,231,235), rgb(250,204,21), rgb(249,115,22), rgb(220,38,38))',
                        }}
                      />
                      <span className="text-slate-500">High</span>
                    </div>

                    {programScoreRange ? (
                      <div className="text-slate-500">
                        Display range: {programScoreRange.displayMin.toFixed(2)} to{' '}
                        {programScoreRange.displayMax.toFixed(2)} (5th–95th percentile clipped)
                      </div>
                    ) : null}
                  </div>
                )}
              </Card>
              
              <div className="grid grid-cols-1 gap-6">
                <Card className="w-full">
                  <div className="flex items-center gap-2 mb-4">
                    <Filter size={18} className="text-slate-400" />
                    <h3 className="font-semibold text-white">Cell Type Filters</h3>
                  </div>
                  {sourceViz.cell_types?.length ? (
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
                        {sourceViz.cell_types.map((cellType) => (
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

                <Card className="w-full overflow-hidden p-0">
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
                      {sourceViz.de_by_cell_type?.length ? (
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
                      <>
                        <div className="flex flex-wrap items-center gap-3 mb-4 text-xs text-slate-300">
                          <div className="flex items-center gap-2">
                            <Filter size={14} className="text-slate-400" />
                            <span>Filters</span>
                          </div>
                          <select
                            value={deFilterType}
                            onChange={(event) => setDeFilterType(event.target.value)}
                            className="bg-slate-900/70 border border-slate-700/70 rounded-lg px-2 py-1 text-xs text-slate-200"
                          >
                            <option value="all">All</option>
                            <option value="ligand">Ligand</option>
                            <option value="receptor">Receptor</option>
                            <option value="druggable">Druggable</option>
                          </select>
                          <input
                            type="text"
                            value={deGeneSearch}
                            onChange={(event) => setDeGeneSearch(event.target.value)}
                            placeholder="Search gene"
                            className="bg-slate-900/70 border border-slate-700/70 rounded-lg px-2 py-1 text-xs text-slate-200"
                          />
                          <div className="flex items-center gap-2">
                            <span className="text-slate-400">Sort</span>
                            <select
                              value={deSortBy}
                              onChange={(event) => setDeSortBy(event.target.value)}
                              className="bg-slate-900/70 border border-slate-700/70 rounded-lg px-2 py-1 text-xs text-slate-200"
                            >
                              <option value="score">Score</option>
                              <option value="logfc">LogFC</option>
                              <option value="padj">Adj. p-value</option>
                              <option value="drug_count">Drug Count</option>
                              <option value="gene">Gene</option>
                            </select>
                            <select
                              value={deSortDir}
                              onChange={(event) => setDeSortDir(event.target.value)}
                              className="bg-slate-900/70 border border-slate-700/70 rounded-lg px-2 py-1 text-xs text-slate-200"
                            >
                              <option value="desc">Desc</option>
                              <option value="asc">Asc</option>
                            </select>
                          </div>
                          <span className="text-slate-500">
                            {deRows.length} shown
                          </span>
                        </div>
                        <div className="overflow-x-auto max-h-[360px]">
                          <table className="w-full text-left text-sm text-slate-300">
                            <thead className="text-slate-400 uppercase text-xs">
                              <tr>
                                <th className="pb-2 pr-4">Gene</th>
                                <th className="pb-2 pr-4">Score</th>
                                <th className="pb-2 pr-4">LogFC</th>
                                <th className="pb-2 pr-4">Adj. p</th>
                                <th className="pb-2 pr-4">Annotations</th>
                                <th className="pb-2 pr-4">Drug Count</th>
                                <th className="pb-2">Drug Names</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-800/70">
                              {deRows.map((row) => {
                                const {
                                  gene,
                                  index,
                                  geneAnnotation,
                                  hasLigand,
                                  hasReceptor,
                                  drugTargets,
                                  drugCount,
                                  padj,
                                } = row;
                                const drugNames = Array.isArray(drugTargets)
                                  ? drugTargets
                                      .map(
                                        (target) => target?.drug_name || target?.drug_claim_name
                                      )
                                      .filter(Boolean)
                                  : [];
                                const uniqueDrugNames = Array.from(
                                  new Set(drugNames.map((name) => String(name)))
                                );
                                const shownDrugNames = uniqueDrugNames.slice(0, 3);
                                const remainingDrugNames =
                                  uniqueDrugNames.length - shownDrugNames.length;
                                return (
                                  <tr key={`${activeDeGroup.group}-${gene}`}>
                                    <td className="py-2 pr-4 font-mono text-xs text-slate-200">
                                      {gene}
                                    </td>
                                    <td className="py-2 pr-4 text-slate-400 text-xs">
                                      {activeDeGroup.scores?.[index] != null &&
                                      !Number.isNaN(activeDeGroup.scores[index])
                                        ? Number(activeDeGroup.scores[index]).toFixed(3)
                                        : '—'}
                                    </td>
                                    <td className="py-2 pr-4 text-slate-400 text-xs">
                                      {activeDeGroup.logfoldchanges?.[index] != null &&
                                      !Number.isNaN(activeDeGroup.logfoldchanges[index])
                                        ? Number(activeDeGroup.logfoldchanges[index]).toFixed(3)
                                        : '—'}
                                    </td>
                                    <td className="py-2 pr-4 text-slate-400 text-xs font-mono">
                                      {formatDePadj(padj)}
                                    </td>
                                    <td className="py-2 pr-4 text-xs text-slate-300">
                                      <div className="flex flex-wrap items-center gap-2">
                                        {hasLigand ? (
                                          <span className="rounded-full bg-emerald-500/15 text-emerald-200 px-2 py-0.5 text-[10px] uppercase">
                                            Ligand
                                          </span>
                                        ) : null}
                                        {hasReceptor ? (
                                          <span className="rounded-full bg-cyan-500/15 text-cyan-200 px-2 py-0.5 text-[10px] uppercase">
                                            Receptor
                                          </span>
                                        ) : null}
                                        {!hasLigand && !hasReceptor && drugCount > 0 ? (
                                          <span className="rounded-full bg-amber-500/15 text-amber-200 px-2 py-0.5 text-[10px] uppercase">
                                            Druggable
                                          </span>
                                        ) : null}
                                      </div>
                                      {!hasLigand && !hasReceptor && !drugCount ? (
                                        <span className="text-slate-500 text-[10px]">—</span>
                                      ) : null}
                                    </td>
                                    <td className="py-2 pr-4 text-slate-400 text-xs">
                                      {drugCount || '—'}
                                    </td>
                                    <td className="py-2 text-slate-400 text-xs">
                                      {drugCount ? (
                                        <>
                                          {shownDrugNames.join(', ')}
                                          {remainingDrugNames > 0
                                            ? ` +${remainingDrugNames} more`
                                            : ''}
                                        </>
                                      ) : (
                                        '—'
                                      )}
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                        {overlapReport?.targets_loaded ? (
                          <p className="mt-4 text-xs text-slate-500">
                            Target sets loaded:{' '}
                            {Object.entries(overlapReport.targets_loaded)
                              .map(([key, count]) => `${key.replace(/_/g, ' ')} (${count})`)
                              .join(' • ')}
                          </p>
                        ) : null}
                      </>
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
          {activeTab === 'electrophysiology' && !ephysVizResults && (
            !ephysShowPreview ? (
              <div className="h-full min-h-[400px] flex flex-col items-center justify-center border-2 border-dashed border-slate-700 rounded-2xl bg-slate-800/20 text-slate-500">
                <Radio size={48} className="mb-4 opacity-50 text-amber-400/70" />
                <p className="text-lg">Traces & spike metrics</p>
                <p className="text-sm opacity-60 text-center max-w-md px-4">
                  Trace preview only. For single-cell, use PREPS + UMAP in the left column, or drop a
                  .h5ad on the upload area above.
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <Card className="text-center p-4">
                    <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                      Spike rate (est.)
                    </p>
                    <p className="text-3xl font-bold text-amber-300">
                      4.2{' '}
                      <span className="text-lg font-normal text-slate-500">Hz</span>
                    </p>
                  </Card>
                  <Card className="text-center p-4">
                    <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                      ISI CV
                    </p>
                    <p className="text-3xl font-bold text-amber-200/90">0.42</p>
                  </Card>
                  <Card className="text-center p-4">
                    <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                      Duration
                    </p>
                    <p className="text-3xl font-bold text-white">
                      124<span className="text-lg font-normal text-slate-500">s</span>
                    </p>
                  </Card>
                </div>
                <Card className="overflow-hidden p-0">
                  <div className="p-4 border-b border-slate-700/50 flex flex-wrap justify-between items-center gap-3 bg-slate-800/50">
                    <div className="flex items-center gap-2">
                      <Activity className="text-amber-400" size={18} />
                      <h3 className="font-semibold text-white">Voltage trace</h3>
                    </div>
                    <span className="text-xs text-slate-500">Placeholder waveform</span>
                  </div>
                  <div className="p-4 bg-slate-900/40">
                    <svg
                      viewBox="0 0 800 200"
                      className="w-full h-[240px] bg-slate-950 rounded-xl border border-slate-800/80"
                      aria-hidden
                    >
                      <line
                        x1="0"
                        y1="100"
                        x2="800"
                        y2="100"
                        stroke="#334155"
                        strokeWidth="1"
                        strokeDasharray="6 6"
                      />
                      <path
                        d="M0,100 C80,30 120,170 160,100 S280,40 360,100 S440,160 520,100 S600,50 680,100 S760,140 800,95"
                        fill="none"
                        stroke="#f59e0b"
                        strokeWidth="2"
                        strokeLinecap="round"
                      />
                    </svg>
                  </div>
                </Card>
                <Card className="p-4">
                  <h4 className="text-sm font-semibold text-white mb-3">Spike raster (preview)</h4>
                  <div className="flex gap-0.5 flex-wrap h-16 items-end rounded-lg bg-slate-900/60 p-2 border border-slate-700/50">
                    {[
                      12, 28, 44, 58, 73, 88, 102, 118, 134, 150, 165, 182, 198, 214, 230, 246,
                      262, 278, 292, 308,
                    ].map((x) => (
                      <div
                        key={x}
                        className="w-0.5 rounded-sm bg-amber-500/80"
                        style={{ height: `${8 + (x % 11)}px` }}
                      />
                    ))}
                  </div>
                  <p className="text-xs text-slate-500 mt-3">
                    Mock raster — replace with detected events from the API.
                  </p>
                </Card>
                <p className="text-xs text-slate-500">
                  Selected file:{' '}
                  <span className="font-mono text-slate-400">{ephysFileName}</span> · Sampling{' '}
                  {ephysSamplingRateKhz} kHz · Bandpass {ephysFilterLowHz}–{ephysFilterHighHz} Hz
                </p>
              </div>
            )
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
          <ChatPage
            embedded
            onClose={() => setShowChat(false)}
            analysisSummaryPath={analysisSummaryPath}
          />
        </Card>
      </div>
    </div>
  );
};

export default PortalPage;

