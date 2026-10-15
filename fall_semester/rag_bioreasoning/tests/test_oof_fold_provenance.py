from __future__ import annotations

import numpy as np
import pandas as pd

from ephys_rag.lgbm_rank import oof_scores


def test_oof_scores_can_return_one_group_safe_fold_per_seed():
    """Catches missing provenance or assigning rows of one gene to different folds."""
    genes = np.repeat([f"G{i}" for i in range(8)], 2)
    y = np.repeat([0, 0, 0, 0, 1, 1, 1, 1], 2)
    X = pd.DataFrame(
        {
            "signal": y.astype(float),
            "noise": np.linspace(-1.0, 1.0, len(y)),
        }
    )

    score, score_sd, folds = oof_scores(
        X,
        y,
        genes,
        seeds=(0, 1),
        n_splits=2,
        return_folds=True,
    )

    assert score.shape == (16,)
    assert score_sd.shape == (16,)
    assert folds.shape == (2, 16)
    assert set(np.unique(folds)) == {0, 1}
    for seed_folds in folds:
        for gene in np.unique(genes):
            assert len(set(seed_folds[genes == gene])) == 1
