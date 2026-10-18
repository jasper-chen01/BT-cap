import React from 'react';

export default function TracePanel({ traces = [], defaultOpen = false }) {
  if (!traces.length) return null;
  return (
    <details open={defaultOpen || undefined} className="mt-4 rounded-xl border border-slate-700/70 bg-slate-950/40 p-3">
      <summary className="cursor-pointer text-sm text-cyan-300">Tool traces ({traces.length})</summary>
      <pre className="mt-3 max-h-96 overflow-auto whitespace-pre-wrap break-words text-xs text-slate-300">{JSON.stringify(traces, null, 2)}</pre>
    </details>
  );
}
