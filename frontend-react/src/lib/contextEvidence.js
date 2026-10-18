// Select returned evidence only; never derive biological scores or rules.
export function contextEvidence(rows, celltype, idh, { pooled = false } = {}) {
  return (rows || []).filter((row) => {
    if (!pooled && row.IDH_status && row.IDH_status !== idh) return false;
    if (row.celltype_id) return row.celltype_id === celltype;
    if (row.source || row.target) return [row.source, row.target].some((value) => value?.split(' / ')[0] === celltype);
    return true;
  });
}
