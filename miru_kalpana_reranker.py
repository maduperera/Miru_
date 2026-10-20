#!/usr/bin/env python3
"""
miru_kalpana_reranker.py
------------------------
Bridge between Miru's retrieval pipeline (BM25 / dense top-K) and the
Kalpana-RIF discrete tensor engine, used as a structural reranker.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

try:  # package re-exports the API at top level
    from kalpana_rif import NeuroVocab, build_tensor
except ImportError:  # fall back to submodule layout
    from kalpana_rif.neuro_vocab import NeuroVocab
    from kalpana_rif.neuro_tensorizer import build_tensor

ASTPair = Tuple[str, str]


class KalpanaReranker:
    """Structural reranker over Miru's top-K candidate chunks."""

    def __init__(self, seed: int = 42) -> None:
        self.vocab = NeuroVocab(seed=seed)

    def index_chunk(self, pairs: List[ASTPair]) -> np.ndarray:
        """Build the structural state tensor for one code chunk."""
        return build_tensor(pairs, self.vocab)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def score_candidates(
        self, query_tensor: np.ndarray, candidate_tensors: Sequence[np.ndarray]
    ) -> List[float]:
        """Cosine similarity of the query against each candidate tensor."""
        return [self._cosine(query_tensor, c) for c in candidate_tensors]

    def rerank(
        self, query_tensor: np.ndarray, candidate_tensors: Sequence[np.ndarray]
    ) -> List[Tuple[int, float]]:
        """Return (candidate_index, score) sorted by descending score."""
        scores = self.score_candidates(query_tensor, candidate_tensors)
        return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    reranker = KalpanaReranker()

    # Roles must be from kalpana_rif.neuro_vocab.ROLES.
    # Target: def authenticate(token): session = validate(...)
    target_pairs = [
        ("ROLE_FUNC", "authenticate"),
        ("ROLE_ARG", "token"),
        ("ROLE_CALL", "validate"),
        ("ROLE_VAR", "session"),
    ]
    # Distractor: identical tokens, roles swapped (bag-of-words twin)
    distractor_pairs = [
        ("ROLE_FUNC", "token"),
        ("ROLE_ARG", "authenticate"),
        ("ROLE_CALL", "session"),
        ("ROLE_VAR", "validate"),
    ]
    # Structural query: "function authenticate taking a token"
    query_pairs = [
        ("ROLE_FUNC", "authenticate"),
        ("ROLE_ARG", "token"),
    ]

    # Miru's BM25 order (distractor first: identical bag of words)
    candidates = {"distractor": distractor_pairs, "target": target_pairs}
    names = list(candidates)
    tensors = [reranker.index_chunk(p) for p in candidates.values()]
    query = reranker.index_chunk(query_pairs)

    print("=" * 52)
    print(" Kalpana-RIF Structural Rerank (Miru top-K)")
    print("=" * 52)
    print(" BM25 input order :", ", ".join(names))
    for rank, (idx, score) in enumerate(reranker.rerank(query, tensors), 1):
        print(f"  #{rank}  {names[idx]:<12} cos = {score:+.4f}")
    print("=" * 52)
