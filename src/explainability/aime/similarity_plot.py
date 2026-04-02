"""Similarity distribution plotting for AIME representative instances."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def save_similarity_plot(aime_state: dict[str, Any], output_path: Path, config: dict[str, Any]) -> Path:
    """Save cosine-similarity histogram against representative instances."""
    X_centered = np.asarray(aime_state["x_centered"], dtype=float)
    rep_indices = config.get("representative_indices", [0])
    rep_indices = [int(i) for i in rep_indices if 0 <= int(i) < X_centered.shape[0]]
    if not rep_indices:
        rep_indices = [0]

    sims: list[float] = []
    for idx in rep_indices:
        rep = X_centered[idx].reshape(1, -1)
        cos = cosine_similarity(X_centered, rep).ravel()
        sims.extend(cos.tolist())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(sims, bins=int(config.get("bins", 30)), color="#2F6DB0", alpha=0.8)
    plt.title("AIME Representative Similarity Distribution")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
