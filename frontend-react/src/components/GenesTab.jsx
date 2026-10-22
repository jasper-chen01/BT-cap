import React, { useEffect, useState } from 'react';
import { Download, Search } from 'lucide-react';
import Card from './ui/Card';
import Button from './ui/Button';
import TracePanel from './TracePanel';
import { askReasoning, getGene, getNeighbors, getRankedGenes, rowsToCsv } from '../lib/reasoningApi';
import { contextEvidence } from '../lib/contextEvidence';

const CELL_TYPES = ['OPC_GABA_like_tumor', 'OPC_GABA_AC_like_tumor', 'cycling_tumor', 'AC_like_tumor', 'T cell', 'TAM1/TAM2/microglia'];
const IDH_CONTEXTS = ['IDH_Mutant', 'IDH_WT'];
const PROGRAMS = { tumor_synaptic: 'Tumor synaptic', immune_synapse: 'Immune synapse' };
const ARMS = { tables: 'Tables only', tables_scetm: 'Tables + scETM', scetm_only: 'scETM only' };
const BADGES = { in_syngo: 'SynGO', in_go_0001772: 'GO:0001772', in_published_33: 'Published-33', housekeeping: 'Housekeeping', passes_three_table: 'Three-table rule' };
const fieldClass = 'w-full rounded-lg border border-slate-700/70 bg-slate-900/70 px-3 py-2 text-sm text-slate-200';
const display = (value) => value == null ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value);

function Select({ label, value, onChange, options }) {
  return <label className="block text-xs text-slate-400">{label}<select className={`${fieldClass} mt-1`} value={value} onChange={(event) => onChange(event.target.value)}>{Object.entries(options).map(([key, text]) => <option key={key} value={key}>{text}</option>)}</select></label>;
}

function ErrorMessage({ error }) {
  return error ? <p role="alert" className="my-3 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-200">{error}</p> : null;
}

function DataTable({ rows, columns = rows.length ? Object.keys(rows[0]) : [] }) {
  if (!rows.length) return <p className="py-3 text-sm text-slate-500">No rows in the tables for this selection.</p>;
  return <div className="max-h-80 overflow-auto"><table className="w-full text-left text-xs text-slate-300"><thead className="sticky top-0 bg-slate-900 text-slate-400"><tr>{columns.map((key) => <th key={key} className="whitespace-nowrap p-2 font-medium">{key.replaceAll('_', ' ')}</th>)}</tr></thead><tbody className="divide-y divide-slate-700/50">{rows.map((row, index) => <tr key={index}>{columns.map((key) => <td key={key} className="p-2 align-top"><span className="block max-w-xs break-words">{display(row[key])}</span></td>)}</tr>)}</tbody></table></div>;
}

export default function GenesTab() {
  const [model, setModel] = useState('tumor_synaptic');
  const [geneInput, setGeneInput] = useState('NRXN1');
  const [gene, setGene] = useState(null);
  const [geneBusy, setGeneBusy] = useState(false);
  const [geneError, setGeneError] = useState('');
  const [celltype, setCelltype] = useState(CELL_TYPES[0]);
  const [idh, setIdh] = useState(IDH_CONTEXTS[0]);
  const [filters, setFilters] = useState({ arm: 'tables_scetm', exclude_housekeeping: true, novel_only: false, rule_pass: '', min_n: 50, limit: 15 });
  const [ranked, setRanked] = useState(null);
  const [rankBusy, setRankBusy] = useState(false);
  const [rankError, setRankError] = useState('');
  const [neighborInput, setNeighborInput] = useState('RPL5');
  const [neighborQuery, setNeighborQuery] = useState(null);
  const [neighbors, setNeighbors] = useState(null);
  const [neighborBusy, setNeighborBusy] = useState(false);
  const [neighborError, setNeighborError] = useState('');
  const [question, setQuestion] = useState('Is NRXN1 DEG-supported as Ephys_2-high in IDH-mutant cycling tumor?');
  const [llm, setLlm] = useState('none');
  const [answer, setAnswer] = useState(null);
  const [askBusy, setAskBusy] = useState(false);
  const [askError, setAskError] = useState('');

  useEffect(() => {
    let cancelled = false;
    setRankBusy(true); setRankError(''); setRanked(null);
    getRankedGenes({ model, ...filters }).then((data) => { if (!cancelled) setRanked(data); })
      .catch((error) => { if (!cancelled) setRankError(error.message); })
      .finally(() => { if (!cancelled) setRankBusy(false); });
    return () => { cancelled = true; };
  }, [model, filters]);

  useEffect(() => {
    if (!neighborQuery) return undefined;
    let cancelled = false;
    setNeighborBusy(true); setNeighborError(''); setNeighbors(null);
    getNeighbors(neighborQuery.symbol, model, 20).then((data) => { if (!cancelled) setNeighbors(data); })
      .catch((error) => { if (!cancelled) setNeighborError(error.message); })
      .finally(() => { if (!cancelled) setNeighborBusy(false); });
    return () => { cancelled = true; };
  }, [neighborQuery, model]);

  async function searchGene(event) {
    event.preventDefault();
    if (!geneInput.trim()) return;
    setGeneBusy(true); setGeneError(''); setGene(null);
    try { setGene(await getGene(geneInput.trim())); }
    catch (error) { setGeneError(error.message); }
    finally { setGeneBusy(false); }
  }

  async function ask(event) {
    event.preventDefault();
    if (!question.trim()) return;
    setAskBusy(true); setAskError(''); setAnswer(null);
    try { setAnswer(await askReasoning(question.trim(), llm)); }
    catch (error) { setAskError(error.message); }
    finally { setAskBusy(false); }
  }

  function download() {
    const url = URL.createObjectURL(new Blob([rowsToCsv(ranked.rows, ranked.columns)], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url; link.download = `genes-${model}-${filters.arm}.csv`; link.click();
    URL.revokeObjectURL(url);
  }

  const updateFilter = (key, value) => setFilters((previous) => ({ ...previous, [key]: value }));
  const contextRows = (gene?.rows || []).filter((row) => row.model === model && row.celltype_id === celltype && row.IDH_status === idh);

  return <div className="mt-8 space-y-6">
    <div className="flex flex-wrap items-end justify-between gap-4"><div><h3 className="text-xl font-bold text-white">Genes & grounded evidence</h3><p className="text-sm text-slate-400">Held-out model outputs and supporting evidence from the shared reasoning tables.</p></div><div className="w-56"><Select label="Program model" value={model} onChange={setModel} options={PROGRAMS} /></div></div>
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 items-start">
      <Card className="min-w-0 xl:col-span-2">
        <h4 className="mb-4 text-lg font-semibold text-white">Gene card</h4>
        <form onSubmit={searchGene} className="flex flex-wrap items-end gap-3"><label className="min-w-40 flex-1 text-xs text-slate-400">Gene symbol<input className={`${fieldClass} mt-1`} value={geneInput} onChange={(event) => setGeneInput(event.target.value)} placeholder="NRXN1" required /></label><Button type="submit" icon={Search} disabled={geneBusy}>{geneBusy ? 'Loading…' : 'Find gene'}</Button></form>
        <ErrorMessage error={geneError} />
        {!gene && !geneBusy && !geneError && <p className="mt-4 text-sm text-slate-500">Search a gene to inspect context scores and evidence.</p>}
        {gene && <>
          <div className="my-5 flex flex-wrap items-center gap-2"><span className="mr-2 font-mono text-xl text-white">{gene.gene}</span>{Object.entries(BADGES).map(([key, label]) => <span key={key} className={`rounded-full px-3 py-1 text-xs ${gene.badges?.[key] ? 'bg-cyan-500/15 text-cyan-200' : 'bg-slate-700/40 text-slate-400'}`}>{label}: {gene.badges?.[key] == null ? 'unavailable' : gene.badges[key] ? 'yes' : 'no'}</span>)}</div>
          <p className="mb-3 text-xs text-slate-400">Select a context for evidence. Each cell shows Tables only / Tables + scETM. Blank means no model output.</p>
          <div className="overflow-x-auto"><table className="w-full text-sm"><thead><tr><th className="p-2 text-left text-slate-400">Cell type</th>{IDH_CONTEXTS.map((context) => <th key={context} className="p-2 text-slate-300">{context.replaceAll('_', ' ')}</th>)}</tr></thead><tbody>{CELL_TYPES.map((cell) => <tr key={cell}><th className="p-2 text-left text-xs font-normal text-slate-300">{cell}</th>{IDH_CONTEXTS.map((context) => <td key={context} className="p-1"><button type="button" onClick={() => { setCelltype(cell); setIdh(context); }} aria-pressed={celltype === cell && idh === context} aria-label={`${cell}, ${context}`} className={`w-full rounded-lg border px-3 py-3 text-xs font-mono text-cyan-200 ${celltype === cell && idh === context ? 'border-cyan-500 bg-cyan-500/10' : 'border-slate-700/60 hover:bg-slate-700/40'}`}><span className="grid grid-cols-2 gap-4">{['tables', 'tables_scetm'].map((arm) => <span key={arm} className="min-h-4" title={ARMS[arm]}>{display(gene.rows.find((row) => row.celltype_id === cell && row.IDH_status === context && row.model === model && row.arm === arm)?.score)}</span>)}</span></button></td>)}</tr>)}</tbody></table></div>
          <div className="mt-6 space-y-4 border-t border-slate-700/50 pt-4"><h5 className="text-sm font-semibold text-white">{celltype} · {idh.replaceAll('_', ' ')} · {PROGRAMS[model]}</h5>
            <DataTable rows={contextRows} columns={['arm', 'score', 'rank_in_context', 'why', 'scoring_folds', 'passes_three_table', 'min_n']} />
            <h5 className="text-sm font-semibold text-slate-200">CellChat edges · {gene.cellchat?.source_file}</h5><p className="text-xs text-slate-500">CellChat evidence is pooled across IDH when the source has no IDH field.</p><DataTable rows={contextEvidence(gene.cellchat?.rows, celltype, idh)} />
            <h5 className="text-sm font-semibold text-slate-200">Context DEG · {gene.deg?.source_file}</h5><DataTable rows={contextEvidence(gene.deg?.rows, celltype, idh)} />
            <details className="text-sm text-slate-400"><summary className="cursor-pointer">Pooled DEG evidence</summary><DataTable rows={contextEvidence(gene.pooled_deg?.rows, celltype, idh, { pooled: true })} /></details>
          </div>
        </>}
      </Card>

      <Card className="min-w-0">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3"><h4 className="text-lg font-semibold text-white">Ranked genes</h4><Button variant="secondary" icon={Download} onClick={download} disabled={!ranked?.rows?.length || rankBusy} className="px-3 py-2 text-xs">Download CSV</Button></div>
        <div className="grid grid-cols-2 gap-3"><Select label="Feature arm" value={filters.arm} onChange={(value) => updateFilter('arm', value)} options={ARMS} /><Select label="Three-table rule" value={String(filters.rule_pass)} onChange={(value) => updateFilter('rule_pass', value === '' ? '' : value === 'true')} options={{ '': 'All', true: 'Pass', false: 'Does not pass' }} />
          <label className="text-xs text-slate-400">Minimum cells<input type="number" min="50" className={`${fieldClass} mt-1`} value={filters.min_n} onChange={(event) => updateFilter('min_n', event.target.value)} /></label><label className="text-xs text-slate-400">Result limit<input type="number" min="0" className={`${fieldClass} mt-1`} value={filters.limit} onChange={(event) => updateFilter('limit', event.target.value)} /></label>
        </div>
        <div className="my-4 flex flex-wrap gap-4 text-sm text-slate-300"><label className="flex items-center gap-2"><input type="checkbox" checked={filters.exclude_housekeeping} onChange={(event) => updateFilter('exclude_housekeeping', event.target.checked)} className="accent-cyan-500" />Exclude housekeeping</label><label className="flex items-center gap-2"><input type="checkbox" checked={filters.novel_only} onChange={(event) => updateFilter('novel_only', event.target.checked)} className="accent-cyan-500" />Novel only</label></div>
        <ErrorMessage error={rankError} />{rankBusy && <p role="status" className="text-sm text-slate-400">Loading rankings…</p>}{ranked && <><p className="mb-2 text-xs text-slate-500">{ranked.rows.length} returned · gene score and rank from the selected arm</p><DataTable rows={ranked.rows} columns={['gene', 'score', 'gene_rank', 'celltype_id', 'IDH_status', 'context_score', 'passes_three_table', 'why', 'scoring_folds']} /></>}
      </Card>

      <Card className="min-w-0">
        <h4 className="mb-4 text-lg font-semibold text-white">Top 20 scETM neighbours</h4>
        <form onSubmit={(event) => { event.preventDefault(); if (neighborInput.trim()) setNeighborQuery({ symbol: neighborInput.trim() }); }} className="flex flex-wrap items-end gap-3"><label className="min-w-32 flex-1 text-xs text-slate-400">Embedding gene<input className={`${fieldClass} mt-1`} value={neighborInput} onChange={(event) => setNeighborInput(event.target.value)} required /></label><Button type="submit" variant="secondary" disabled={neighborBusy}>Find neighbours</Button></form>
        <p className="my-3 text-xs text-slate-400">Cosine similarity and label membership for {PROGRAMS[model]}.</p><ErrorMessage error={neighborError} />{neighborBusy && <p role="status" className="text-sm text-slate-400">Loading neighbours…</p>}{neighbors && <><p className="mb-2 font-mono text-sm text-cyan-300">{neighbors.gene}</p><DataTable rows={neighbors.rows} columns={['gene', 'similarity', 'in_label_set']} /></>}
      </Card>

      <Card className="min-w-0 xl:col-span-2">
        <h4 className="mb-4 text-lg font-semibold text-white">Ask the tables</h4>
        <form onSubmit={ask} className="space-y-3"><label className="block text-xs text-slate-400">Biological question<textarea className={`${fieldClass} mt-1`} rows={3} value={question} onChange={(event) => setQuestion(event.target.value)} required /></label><div className="flex flex-wrap items-end gap-3"><div className="w-48"><Select label="Answer model" value={llm} onChange={setLlm} options={{ none: 'No LLM', medgemma: 'MedGemma', gemini: 'Gemini' }} /></div><Button type="submit" disabled={askBusy}>{askBusy ? 'Looking up evidence…' : 'Ask'}</Button></div></form>
        <ErrorMessage error={askError} />{answer && <div className="mt-5"><p className="mb-3 text-xs text-slate-500">{answer.model} · {answer.cached ? 'Cached response' : 'Response'}</p><div className="whitespace-pre-wrap break-words text-sm leading-relaxed text-slate-200">{answer.answer}</div>{!!answer.hits?.length && <div className="mt-4"><h5 className="text-xs font-semibold text-slate-400">Retrieved evidence</h5><ul className="mt-2 space-y-1 text-xs text-slate-300">{answer.hits.map((hit, index) => <li key={index}>{hit.title} {hit.source_file && <span className="text-slate-500">· {hit.source_file}</span>}</li>)}</ul></div>}<TracePanel traces={answer.traces} /></div>}
      </Card>
    </div>
  </div>;
}
