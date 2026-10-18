export const API_BASE = (import.meta.env?.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '');

export async function apiRequest(path, options) {
  const response = await fetch(`${API_BASE}/api${path}`, options);
  const data = await response.json();
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return data;
}

export const getGene = (symbol) => apiRequest(`/genes/${encodeURIComponent(symbol)}`);

export function getRankedGenes(filters = {}) {
  const values = { model: 'tumor_synaptic', arm: 'tables_scetm', exclude_housekeeping: true,
    novel_only: false, min_n: 0, limit: 15, ...filters };
  const query = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== '') query.set(key, value);
  });
  return apiRequest(`/genes/ranked?${query}`);
}

export const getNeighbors = (symbol, model = 'tumor_synaptic', k = 20) =>
  apiRequest(`/genes/${encodeURIComponent(symbol)}/neighbors?${new URLSearchParams({ model, k })}`);

export const askReasoning = (question, model = 'none') => apiRequest('/reasoning/ask', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, model }),
});

export function rowsToCsv(rows, columns = rows.length ? Object.keys(rows[0]) : []) {
  const escape = (value) => {
    const text = value == null ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value);
    return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  if (!columns.length) return '';
  return [columns.map(escape).join(','), ...rows.map((row) => columns.map((column) => escape(row[column])).join(','))].join('\r\n');
}

export function normalizeChatMessage(message = {}) {
  return { ...message, id: message.id ?? globalThis.crypto.randomUUID(),
    role: message.role ?? 'assistant', content: message.content ?? '', traces: message.traces ?? [] };
}
