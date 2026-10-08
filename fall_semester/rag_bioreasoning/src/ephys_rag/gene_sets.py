"""Knowledge-tool gene sets used as LightGBM labels.

tumor_synaptic  <- SynGO (expert-curated synaptic annotations, human HGNC orthologs)
immune_synapse  <- QuickGO GO:0001772 "immunological synapse" (+ descendants), all taxa,
                   symbols upper-cased to match human gene symbols

`fetch_all()` downloads both and writes one TSV per set to knowledge/gene_sets/, plus
SOURCES.json with URLs, release and counts. The TSVs are committed so runs are reproducible.
"""
from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ephys_rag.config import KNOWLEDGE_DIR

GENE_SET_DIR = KNOWLEDGE_DIR / "gene_sets"
SOURCES_JSON = GENE_SET_DIR / "SOURCES.json"

SYNGO_URL = "https://www.syngoportal.org/data/SynGO_bulk_20250218.json"
QUICKGO_URL = (
    "https://www.ebi.ac.uk/QuickGO/services/annotation/downloadSearch?"
    "goId={go_id}&goUsage=descendants&goUsageRelationships=is_a,part_of,occurs_in&downloadLimit=50000"
)
EXPERIMENTAL_CODES = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"}
_UA = {"User-Agent": "Mozilla/5.0 (ephys-rag gene-set fetcher)"}
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-]*$")


@dataclass(frozen=True)
class GeneSetSpec:
    name: str
    filename: str
    label_compartments: tuple[str, ...]
    description: str


GENE_SETS: dict[str, GeneSetSpec] = {
    "tumor_synaptic": GeneSetSpec(
        name="tumor_synaptic",
        filename="tumor_synaptic_syngo.tsv",
        label_compartments=("tumor",),
        description="SynGO synaptic genes, scored in tumor cell types (tumor synaptic program)",
    ),
    "immune_synapse": GeneSetSpec(
        name="immune_synapse",
        filename="immune_synapse_go0001772.tsv",
        label_compartments=("T_cell", "TAM_microglia", "tumor"),
        description="GO:0001772 immunological synapse genes, scored on both sides of the synapse",
    ),
}


def _get(url: str, accept: str | None = None, timeout: int = 180) -> bytes:
    headers = dict(_UA)
    if accept:
        headers["Accept"] = accept
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout).read()


def fetch_syngo(url: str = SYNGO_URL) -> tuple[pd.DataFrame, dict]:
    payload = json.loads(_get(url))
    domains = {term["id"]: term.get("goDomain", "") for term in payload.get("goterms", [])}
    per_gene: dict[str, dict] = defaultdict(lambda: {"terms": set(), "pmids": set(), "domains": Counter(), "n": 0})
    for ann in payload.get("annotations", []):
        for ortholog in ann.get("orthologs") or []:
            symbol = (ortholog.get("hgnc_symbol") or "").strip()
            if not symbol:
                continue
            row = per_gene[symbol]
            row["n"] += 1
            row["terms"].add(ann.get("goterm", ""))
            row["pmids"].add(str(ann.get("pmid", "")))
            row["domains"][domains.get(ann.get("goterm", ""), "")] += 1
    frame = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "n_annotations": row["n"],
                "n_cc": row["domains"].get("CC", 0),
                "n_bp": row["domains"].get("BP", 0),
                "n_papers": len(row["pmids"]),
                "go_terms": ";".join(sorted(t for t in row["terms"] if t)),
            }
            for symbol, row in per_gene.items()
        ]
    ).sort_values(["n_annotations", "symbol"], ascending=[False, True])
    meta = {
        "source": "SynGO",
        "url": url,
        "release": payload.get("release"),
        "n_annotations": len(payload.get("annotations", [])),
        "n_genes": int(len(frame)),
    }
    return frame, meta


def fetch_quickgo(go_id: str = "GO:0001772") -> tuple[pd.DataFrame, dict]:
    url = QUICKGO_URL.format(go_id=go_id)
    lines = _get(url, accept="text/tsv").decode("utf-8").splitlines()
    header = lines[0].split("\t")
    rows = [dict(zip(header, line.split("\t"))) for line in lines[1:] if line.strip()]
    per_gene: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "n_human": 0, "taxa": set(), "codes": Counter(), "experimental": 0}
    )
    skipped = 0
    for rec in rows:
        symbol = rec["SYMBOL"].strip().upper()
        if not _SYMBOL.match(symbol) or symbol == rec["GENE PRODUCT ID"].upper():
            skipped += 1
            continue
        row = per_gene[symbol]
        code = rec["GO EVIDENCE CODE"]
        row["n"] += 1
        row["taxa"].add(rec["TAXON ID"])
        row["codes"][code] += 1
        row["n_human"] += rec["TAXON ID"] == "9606"
        row["experimental"] += code in EXPERIMENTAL_CODES
    frame = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "n_annotations": row["n"],
                "n_human_annotations": row["n_human"],
                "n_experimental": row["experimental"],
                "n_taxa": len(row["taxa"]),
                "evidence_codes": ";".join(f"{c}:{n}" for c, n in row["codes"].most_common()),
            }
            for symbol, row in per_gene.items()
        ]
    ).sort_values(["n_human_annotations", "n_experimental", "n_annotations"], ascending=False)
    meta = {
        "source": "QuickGO",
        "go_id": go_id,
        "url": url,
        "n_annotations": len(rows),
        "n_annotations_skipped_no_symbol": skipped,
        "n_genes": int(len(frame)),
        "n_genes_human_annotated": int((frame["n_human_annotations"] > 0).sum()),
        "n_genes_experimental": int((frame["n_experimental"] > 0).sum()),
    }
    return frame, meta


def fetch_all(out_dir: Path = GENE_SET_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    syngo, syngo_meta = fetch_syngo()
    immune, immune_meta = fetch_quickgo("GO:0001772")
    syngo.to_csv(out_dir / GENE_SETS["tumor_synaptic"].filename, sep="\t", index=False)
    immune.to_csv(out_dir / GENE_SETS["immune_synapse"].filename, sep="\t", index=False)
    sources = {
        "fetched": date.today().isoformat(),
        "tumor_synaptic": {**syngo_meta, "file": GENE_SETS["tumor_synaptic"].filename},
        "immune_synapse": {**immune_meta, "file": GENE_SETS["immune_synapse"].filename},
    }
    (out_dir / SOURCES_JSON.name).write_text(json.dumps(sources, indent=2), encoding="utf-8")
    return sources


def load_gene_set(name: str, human_only: bool = False, directory: Path = GENE_SET_DIR) -> set[str]:
    """Symbols in a gene set. `human_only` keeps genes with >=1 human annotation (QuickGO sets)."""
    spec = GENE_SETS[name]
    path = directory / spec.filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `ephys-rag fetch-genesets` first.")
    frame = pd.read_csv(path, sep="\t")
    if human_only and "n_human_annotations" in frame.columns:
        frame = frame[frame["n_human_annotations"] > 0]
    return set(frame["symbol"].astype(str))
