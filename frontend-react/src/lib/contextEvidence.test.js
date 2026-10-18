import test from 'node:test';
import assert from 'node:assert/strict';
import { contextEvidence } from './contextEvidence.js';

test('pooled DEG evidence retains the selected cell type in either IDH context', () => {
  const selected = { gene: 'NRXN1', celltype_id: 'OPC_GABA_like_tumor', IDH_status: 'pooled', avg_log2FC: 1.25 };
  const other = { gene: 'NRXN1', celltype_id: 'cycling_tumor', IDH_status: 'pooled', avg_log2FC: -0.5 };
  assert.deepEqual(contextEvidence([selected, other], 'OPC_GABA_like_tumor', 'IDH_Mutant', { pooled: true }), [selected]);
  assert.deepEqual(contextEvidence([selected, other], 'OPC_GABA_like_tumor', 'IDH_WT', { pooled: true }), [selected]);
});

test('within-IDH DEG evidence still restricts both IDH and cell type', () => {
  const selected = { celltype_id: 'OPC_GABA_like_tumor', IDH_status: 'IDH_Mutant' };
  const rows = [selected, { celltype_id: 'OPC_GABA_like_tumor', IDH_status: 'IDH_WT' },
    { celltype_id: 'cycling_tumor', IDH_status: 'IDH_Mutant' },
    { celltype_id: 'OPC_GABA_like_tumor', IDH_status: 'pooled' }];
  assert.deepEqual(contextEvidence(rows, 'OPC_GABA_like_tumor', 'IDH_Mutant'), [selected]);
});
