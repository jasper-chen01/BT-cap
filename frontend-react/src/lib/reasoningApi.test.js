import test from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';
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
  assert.equal(calls[1].url, 'http://127.0.0.1:8000/api/genes/ranked?model=tumor_synaptic&arm=tables_scetm&exclude_housekeeping=true&novel_only=false&min_n=50&limit=15');
  await getRankedGenes({ model: 'immune_synapse', arm: 'tables', exclude_housekeeping: false, novel_only: true, rule_pass: false, min_n: 70, limit: 10 });
  assert.deepEqual(Object.fromEntries(new URL(calls[2].url).searchParams), { model: 'immune_synapse', arm: 'tables', exclude_housekeeping: 'false', novel_only: 'true', rule_pass: 'false', min_n: '70', limit: '10' });
  await getNeighbors('A/B');
  assert.equal(calls[3].url, 'http://127.0.0.1:8000/api/genes/A%2FB/neighbors?model=tumor_synaptic&k=20');
  await askReasoning('NRXN1 evidence?', 'medgemma');
  assert.equal(calls[4].options.method, 'POST');
  assert.deepEqual(JSON.parse(calls[4].options.body), { question: 'NRXN1 evidence?', model: 'medgemma' });
});

test('Genes tab starts ranked-list filtering at the 50-cell safety floor', async () => {
  const server = await createServer({ root: fileURLToPath(new URL('../../', import.meta.url)),
    server: { middlewareMode: true }, appType: 'custom' });
  try {
    const { default: GenesTab } = await server.ssrLoadModule('/src/components/GenesTab.jsx');
    const markup = renderToStaticMarkup(React.createElement(GenesTab));
    assert.match(markup, /Minimum cells<input[^>]*min="50"[^>]*value="50"/);
  } finally {
    await server.close();
  }
});

test('Genes tab starts Ask the Tables with a prompt backed by the MedGemma cache', async () => {
  const server = await createServer({ root: fileURLToPath(new URL('../../', import.meta.url)),
    server: { middlewareMode: true }, appType: 'custom' });
  try {
    const { default: GenesTab } = await server.ssrLoadModule('/src/components/GenesTab.jsx');
    const markup = renderToStaticMarkup(React.createElement(GenesTab));
    assert.match(markup,
      /<textarea[^>]*>Is NRXN1 DEG-supported as Ephys_2-high in IDH-mutant cycling tumor\?<\/textarea>/);
  } finally {
    await server.close();
  }
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
  assert.deepEqual(normalizeChatMessage({ role: 'user', content: 'Question', traces: null }).traces, []);
});

test('chat assistant messages show collapsed traces while ordinary and user messages do not', async () => {
  const server = await createServer({ root: fileURLToPath(new URL('../../', import.meta.url)),
    server: { middlewareMode: true }, appType: 'custom' });
  try {
    const { ChatMessage } = await server.ssrLoadModule('/src/components/ChatPage.jsx');
    const traces = [{ tool: 'nominate_lookup', source_file: 'tables.csv', n_hits: 1,
      filters: { gene: 'NRXN1' }, rows: [{ gene: 'NRXN1', score: null }], note: 'held-out' }];
    const render = (message) => renderToStaticMarkup(React.createElement(ChatMessage, {
      message: normalizeChatMessage(message),
    }));
    const assistant = render({ role: 'assistant', content: 'Grounded answer', traces });
    assert.match(assistant, /Grounded answer/);
    assert.match(assistant, /Tool traces \(1\)/);
    assert.match(assistant, /nominate_lookup/);
    assert.doesNotMatch(assistant, /<details[^>]*\bopen(?:=|\s|>)/);
    assert.doesNotMatch(render({ role: 'assistant', content: 'Hello' }), /<details/);
    assert.doesNotMatch(render({ role: 'user', content: 'Question', traces }), /<details/);
  } finally {
    await server.close();
  }
});
