import { useEffect, useMemo, useState } from 'react';

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

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        if (!data.reference_data_loaded || !data.embeddings_indexed) {
          setStatus({
            type: 'error',
            message:
              'Warning: Reference data or embeddings not loaded. Please run prepare_reference_embeddings.py first.',
          });
        }
      } catch (error) {
        setStatus({
          type: 'error',
          message:
            'Warning: Could not connect to API. Make sure the backend server is running.',
        });
      }
    };

    checkHealth();
  }, []);

  const annotationStats = useMemo(() => {
    if (!results?.annotations?.length) {
      return null;
    }

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
  };

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0] || null;
    setFile(selectedFile);
    setFileName(selectedFile ? `Selected: ${selectedFile.name}` : '');
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const droppedFile = event.dataTransfer.files?.[0] || null;
    if (droppedFile && droppedFile.name.endsWith('.h5ad')) {
      setFile(droppedFile);
      setFileName(`Selected: ${droppedFile.name}`);
    } else {
      showStatus('Please upload a .h5ad file', 'error');
    }
  };

  const annotateData = async () => {
    if (!file) {
      showStatus('Please select a file first', 'error');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('top_k', topK);
    formData.append('similarity_threshold', threshold);

    setIsAnnotating(true);
    showStatus('Processing your data...', 'info');

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
      showStatus(`Successfully annotated ${data.total_cells} cells!`, 'success');
    } catch (error) {
      showStatus(`Error: ${error.message}`, 'error');
      console.error('Annotation error:', error);
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

  const renderResultsTable = () => {
    if (!results) return null;

    const displayCells = results.annotations.slice(0, 100);
    return (
      <>
        {displayCells.map((cell) => {
          const matchesText = cell.top_matches
            .slice(0, 3)
            .map(
              (match) =>
                `${match.annotation} (${(match.similarity * 100).toFixed(1)}%)`
            )
            .join(', ');
          return (
            <tr key={cell.cell_id}>
              <td>{cell.cell_id}</td>
              <td>{cell.predicted_annotation}</td>
              <td>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{
                      width: `${cell.confidence_score * 100}%`,
                    }}
                  />
                </div>{' '}
                {(cell.confidence_score * 100).toFixed(1)}%
              </td>
              <td>{matchesText || 'No matches'}</td>
            </tr>
          );
        })}
        {results.annotations.length > 100 && (
          <tr>
            <td colSpan={4} className="results-more">
              ... and {results.annotations.length - 100} more cells
            </td>
          </tr>
        )}
      </>
    );
  };

  return (
    <div className="portal-page">
      <div className="portal-container">
        <div className="portal-header">
          <div>
            <h1>🧠 Brain Tumor Annotation Portal</h1>
            <p>Annotate glioma single-cell RNA-seq data using vector embeddings</p>
          </div>
          <a href="#/chat" className="portal-link-button">
            💬 Try AI Agent
          </a>
        </div>

        <div className="portal-content">
          <div
            className={`upload-section${isDragging ? ' dragover' : ''}`}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            <h2>Upload Single-Cell Data</h2>
            <p>Drag and drop your h5ad file here, or click to browse</p>
            <div className="file-input-wrapper">
              <input type="file" accept=".h5ad" onChange={handleFileChange} />
              <span className="file-input-button">Choose File</span>
            </div>
            <p className="file-name">{fileName}</p>
          </div>

          <div className="parameters">
            <div className="parameter-group">
              <label htmlFor="topK">Top K Neighbors</label>
              <input
                id="topK"
                type="number"
                value={topK}
                min={1}
                max={50}
                onChange={(event) => setTopK(Number(event.target.value))}
              />
            </div>
            <div className="parameter-group">
              <label htmlFor="threshold">Similarity Threshold</label>
              <input
                id="threshold"
                type="number"
                value={threshold}
                min={0}
                max={1}
                step={0.1}
                onChange={(event) => setThreshold(Number(event.target.value))}
              />
            </div>
          </div>

          <button
            className="annotate-button"
            onClick={annotateData}
            disabled={isAnnotating}
          >
            {isAnnotating ? (
              <>
                Annotating... <span className="loading" />
              </>
            ) : (
              'Annotate Cells'
            )}
          </button>

          {status && (
            <div className={`status ${status.type}`}>{status.message}</div>
          )}

          {results && (
            <div className="results">
              <div className="results-header">
                <h2>Annotation Results</h2>
                <button className="download-button" onClick={downloadResults}>
                  Download Results
                </button>
              </div>

              {annotationStats && (
                <div className="stats">
                  <div className="stat-card">
                    <h3>{annotationStats.totalCells}</h3>
                    <p>Total Cells</p>
                  </div>
                  <div className="stat-card">
                    <h3>{annotationStats.uniqueAnnotations}</h3>
                    <p>Unique Annotations</p>
                  </div>
                  <div className="stat-card">
                    <h3>{(annotationStats.avgConfidence * 100).toFixed(1)}%</h3>
                    <p>Average Confidence</p>
                  </div>
                </div>
              )}

              <div className="results-table-wrapper">
                <table className="results-table">
                  <thead>
                    <tr>
                      <th>Cell ID</th>
                      <th>Predicted Annotation</th>
                      <th>Confidence</th>
                      <th>Top Matches</th>
                    </tr>
                  </thead>
                  <tbody>{renderResultsTable()}</tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PortalPage;

