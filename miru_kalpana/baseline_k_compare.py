#!/usr/bin/env python3
"""
baseline_k_compare.py
---------------------
Miru baseline at K=3 / K=5 vs Miru + Kalpana fused (top-3 of the K=25 pool).

    python3 miru_kalpana/baseline_k_compare.py

Two baseline readings are reported because Miru's requested topK changes its ranking
(see benchmark/results.minilm-l6-256.json):
  * "sliced"  = first 3 / 5 of Miru's topK=25 pool (the pool Kalpana reranks)
  * "native"  = Miru called with topK=3 / topK=5 directly (if the dumps exist)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_kalpana_fusion import defines, in_defined  # noqa: E402

TAG = "minilm-l6-256"
spec = {q["id"]: q for q in json.loads((ROOT / "benchmark" / "queries.topk.json").read_text())["queries"]}
res = json.loads((ROOT / "benchmark" / f"results.kalpana.{TAG}.k25.json").read_text())


def at_n(pools: dict, n: int) -> tuple[float, float, int]:
    f = d = 0
    for row in pools["queries"]:
        q, top = spec[row["id"]], row["candidates"][:n]
        fh = [in_defined(c["file_path"], q["definedIn"]) for c in top]
        f += any(fh)
        d += any(h and defines(c["content"], q["identifier"]) for h, c in zip(fh, top))
    m = len(pools["queries"])
    return f / m, d / m, m


def load(k: int):
    p = ROOT / ".bench" / "kalpana" / f"candidates.{TAG}.k{k}.json"
    return json.loads(p.read_text()) if p.exists() else None


k25 = load(25)
rows = []
for n in (3, 5):
    f, d, m = at_n(k25, n)
    rows.append((f"Miru baseline, top-{n} of topK=25 pool", n, f, d, m))
for n in (3, 5):
    pools = load(n)
    if pools:
        f, d, m = at_n(pools, n)
        rows.append((f"Miru baseline, native topK={n}", n, f, d, m))
cv, rrf = res["arms"]["cv"], res["arms"]["rrf"]
rows.append(("Miru + Kalpana, held-out alpha (top-3 of 25)", 3, cv["hit3"], cv["def3"], res["queries"]))
rows.append(("Miru + Kalpana, RRF (top-3 of 25)", 3, rrf["hit3"], rrf["def3"], res["queries"]))

W = 86
print("=" * W)
print(f" {'Configuration':<48}{'N':>3}{'File@N':>11}{'Chunk@N':>11}{'q':>6}")
print("-" * W)
for label, n, f, d, m in rows:
    print(f" {label:<48}{n:>3}{f:>11.1%}{d:>11.1%}{m:>6}")
print("=" * W)
print(" File@N  = a file defining the identifier is in the top N")
print(" Chunk@N = the chunk containing the definition itself is in the top N")
