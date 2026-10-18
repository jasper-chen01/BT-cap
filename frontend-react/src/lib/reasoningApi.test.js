import test from 'node:test';
import assert from 'node:assert/strict';
import { getGene, getRankedGenes, getNeighbors, askReasoning, rowsToCsv, normalizeChatMessage } from './reasoningApi.js';

test('API paths encode gene symbols and send endpoint filter defaults', async (t) => {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push({ url, options });
    return { ok: true, json: async () => ({ rows: [{ score: null }, { score: '' }, { score: 0 }] }) };
  });
  assert.deepEqual((await getGene('A/B +')).rows, [{ score: null }, { score: '' }, { score: 0 }]);
  assert.equal(calls[0].url, 'http://127.0.0.1:8000/api/genes/A%2FB%20%2B');
  await getRankedGenes();
  assert.equal(calls[1].url, 'http://127.0.0.1:8000/api/genes/ranked?model=tumor_synaptic&arm=tables_scetm&exclude_housekeeping=true&novel_only=false&min_n=0&limit=15');
  await getRankedGenes({ model: 'immune_synapse', arm: 'tables', exclude_housekeeping: false, novel_only: true, rule_pass: false, min_n: 20, limit: 10 });
  assert.deepEqual(Object.fromEntries(new URL(calls[2].url).searchParams), { model: 'immune_synapse', arm: 'tables', exclude_housekeeping: 'false', novel_only: 'true', rule_pass: 'false', min_n: '20', limit: '10' });
  await getNeighbors('A/B');
  assert.equal(calls[3].url, 'http://127.0.0.1:8000/api/genes/A%2FB/neighbors?model=tumor_synaptic&k=20');
  await askReasoning('NRXN1 evidence?', 'medgemma');
  assert.equal(calls[4].options.method, 'POST');
  assert.deepEqual(JSON.parse(calls[4].options.body), { question: 'NRXN1 evidence?', model: 'medgemma' });
});

test('API errors surface backend preparation and validation details', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => ({ ok: false, status: 503, json: async () => ({ detail: 'Runtime unavailable: prepare data' }) }));
  await assert.rejects(getGene('NRXN1'), /Runtime unavailable: prepare data/);
});

test('CSV retains endpoint column order, blank values, zero, and escaped fields', () => {
  const rows = [{ gene: 'A,B', score: null, why: 'one "quote"\nnext' }, { gene: 'B', score: 0, why: '' }];
  assert.equal(rowsToCsv(rows), 'gene,score,why\r\n"A,B",,"one ""quote""\nnext"\r\nB,0,');
  assert.equal(rowsToCsv([{ gene: 'A', score: '' }, { gene: 'B', score: undefined }]), 'gene,score\r\nA,\r\nB,');
  assert.equal(rowsToCsv([]), '');
  assert.equal(rowsToCsv([{ score: null, gene: 'NRXN1' }], ['gene', 'score']), 'gene,score\r\nNRXN1,');
  assert.equal(rowsToCsv([], ['gene', 'score']), 'gene,score');
});

test('message normalization preserves supplied fields and trace arrays', () => {
  const traces = [{ tool: 'nominate_lookup', rows: [{ score: null }] }];
  const message = normalizeChatMessage({ id: 7, role: 'assistant', content: 'Answer', traces });
  assert.deepEqual(message, { id: 7, role: 'assistant', content: 'Answer', traces });
  assert.equal(normalizeChatMessage({ content: 'Hello' }).role, 'assistant');
  assert.deepEqual(normalizeChatMessage({ content: 'Hello' }).traces, []);
});
