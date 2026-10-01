from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

from ephys_rag.schema import Interaction


@dataclass
class PathwayContrast:
    pathway: str
    compartment: str
    ephys_1: int
    ephys_2: int
    delta_e2_minus_e1: int
    mean_prob_e1: float
    mean_prob_e2: float


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pathway_contrasts(
    interactions: Iterable[Interaction],
    *,
    as_source: bool = True,
) -> list[PathwayContrast]:
    counts: dict[tuple[str, str, str], int] = Counter()
    probs: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in interactions:
        compartment = row.source_compartment if as_source else row.target_compartment
        ephys = row.source_ephys if as_source else row.target_ephys
        key = (row.pathway, compartment, ephys)
        counts[key] += 1
        probs[key].append(row.probability)

    pathways = sorted({k[0] for k in counts})
    compartments = sorted({k[1] for k in counts})
    out: list[PathwayContrast] = []
    for pathway in pathways:
        for compartment in compartments:
            e1 = counts.get((pathway, compartment, "Ephys_1"), 0)
            e2 = counts.get((pathway, compartment, "Ephys_2"), 0)
            if e1 == 0 and e2 == 0:
                continue
            out.append(
                PathwayContrast(
                    pathway=pathway,
                    compartment=compartment,
                    ephys_1=e1,
                    ephys_2=e2,
                    delta_e2_minus_e1=e2 - e1,
                    mean_prob_e1=_mean(probs.get((pathway, compartment, "Ephys_1"), [])),
                    mean_prob_e2=_mean(probs.get((pathway, compartment, "Ephys_2"), [])),
                )
            )
    out.sort(key=lambda item: (-abs(item.delta_e2_minus_e1), -(item.ephys_1 + item.ephys_2)))
    return out


def exclusive_pathways(
    contrasts: list[PathwayContrast],
    *,
    compartment: str,
    ephys: str,
) -> list[PathwayContrast]:
    want_e2 = ephys == "Ephys_2"
    hits = []
    for item in contrasts:
        if item.compartment != compartment:
            continue
        if want_e2 and item.ephys_1 == 0 and item.ephys_2 > 0:
            hits.append(item)
        if not want_e2 and item.ephys_2 == 0 and item.ephys_1 > 0:
            hits.append(item)
    return hits


def flow_counts(interactions: Iterable[Interaction]) -> dict[str, int]:
    return dict(Counter(row.flow for row in interactions))


def summarize_dataset(interactions: list[Interaction]) -> dict:
    low_n = [
        row.interaction_name
        for row in interactions
        if (row.source_n_cells is not None and row.source_n_cells < 50)
        or (row.target_n_cells is not None and row.target_n_cells < 50)
    ]
    return {
        "n_interactions": len(interactions),
        "n_pathways": len({row.pathway for row in interactions}),
        "n_pairs": len({row.interaction_name for row in interactions}),
        "source_ephys": dict(Counter(row.source_ephys for row in interactions)),
        "target_ephys": dict(Counter(row.target_ephys for row in interactions)),
        "flows": flow_counts(interactions),
        "n_pairs_with_group_n_lt_50": len(set(low_n)),
        "labeled_identities": sorted(
            {row.source_cell for row in interactions} | {row.target_cell for row in interactions}
        ),
    }

