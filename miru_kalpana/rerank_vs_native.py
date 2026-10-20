#!/usr/bin/env python3
"""
rerank_vs_native.py
-------------------
Paired comparison of a Kalpana-reranked pool's Top-3 against
  (a) Miru's *native* topK=3 Top-3            -> "does it beat Miru's best mode?"
  (b) the same pool's un-reranked first 3     -> "does it ever remove a correct chunk?" (do-no-harm)

    python3 miru_kalpana/rerank_vs_native.py [--pool k5] [--set all500]

Reads benchmark/results.kalpana.<tag>[.<set>].<pool>.json and
.bench/kalpana/candidates.<tag>[.<set>].k3.json. With --set, results are split by query-id prefix:
S* = frozen 150 (used during development), N* = new queries (confirmatory, never seen in development).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_kalpana_fusion import defines, in_defined, sign_test  # noqa: E402

TAG = "minilm-l6-256"
ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="k5")
ap.add_argument("--set", default="", help="query-set tag, e.g. all500 (empty = frozen 150)")
args = ap.parse_args()

st = f".{args.set}" if args.set else ""
qfile = ROOT / "benchmark" / (f"queries.topk.{args.set}.json" if args.set else "queries.topk.json")
spec = {q["id"]: q for q in json.loads(qfile.read_text())["queries"]}
res = json.loads((ROOT / "benchmark" / f"results.kalpana.{TAG}{st}.{args.pool}.json").read_text())
nat = {r["id"]: r["candidates"][:3]
       for r in json.loads((ROOT / ".bench" / "kalpana" / f"candidates.{TAG}{st}.k3.json").read_text())["queries"]}


def native(qid: str) -> dict:
    q, top = spec[qid], nat[qid]
    fh = [in_defined(c["file_path"], q["definedIn"]) for c in top]
    return {"hit3": any(fh), "def3": any(h and defines(c["content"], q["identifier"]) for h, c in zip(fh, top))}


pq = res["perQuery"]
ids = list(range(len(pq)))
random.Random(res["cv"]["seed"]).shuffle(ids)
F = res["cv"]["folds"]
alpha_of = {i: res["cv"]["alphas"][f] for f in range(F) for i in ids[f::F]}

ARMS = {
    "Kalpana RRF": lambda i, r: r["rrf"],
    "Kalpana held-out alpha": lambda i, r: r[f"lin{alpha_of[i]}"],
}


def wl(rows, key):
    w = sum(a[key] and not b[key] for a, b in rows)
    l = sum(b[key] and not a[key] for a, b in rows)
    return w, l, sign_test(w, l)


def report(title: str, sel: list) -> None:
    n = len(sel)
    if not n:
        return
    W = 96
    ceiling = sum(pq[i]["baseline"]["rr_def"] > 0 for i in sel) / n
    nat_rows = [native(pq[i]["id"]) for i in sel]
    print("=" * W)
    print(f" {title}   ({n} queries)   ceiling: defining chunk anywhere in top-{args.pool[1:]} = {ceiling:.1%}")
    print("-" * W)
    print(f" {'Top-3 drawn from':<38}{'File@3':>8}{'Chunk@3':>9}"
          f"{'vs native K=3: Chunk W/L':>27}{'p':>8}{'File W/L':>10}{'p':>7}")
    print(f" {'Miru native topK=3':<38}{sum(r['hit3'] for r in nat_rows)/n:>8.1%}"
          f"{sum(r['def3'] for r in nat_rows)/n:>9.1%}{'—':>27}")
    base = [pq[i]["baseline"] for i in sel]
    print(f" {'Miru topK=' + args.pool[1:] + ' first 3 (no rerank)':<38}{sum(r['hit3'] for r in base)/n:>8.1%}"
          f"{sum(r['def3'] for r in base)/n:>9.1%}", end="")
    w, l, p = wl(list(zip(base, nat_rows)), "def3"); fw, fl, fp = wl(list(zip(base, nat_rows)), "hit3")
    print(f"{w:>21}/{l:<5}{p:>8.4f}{fw:>6}/{fl:<3}{fp:>7.3f}")
    for name, get in ARMS.items():
        rows = [get(i, pq[i]) for i in sel]
        w, l, p = wl(list(zip(rows, nat_rows)), "def3"); fw, fl, fp = wl(list(zip(rows, nat_rows)), "hit3")
        print(f" {name:<38}{sum(r['hit3'] for r in rows)/n:>8.1%}{sum(r['def3'] for r in rows)/n:>9.1%}"
              f"{w:>21}/{l:<5}{p:>8.4f}{fw:>6}/{fl:<3}{fp:>7.3f}")
    print("-" * W)
    print(f" Do-no-harm vs the same top-{args.pool[1:]} pool un-reranked (correct defining chunk removed from top 3 = loss):")
    for name, get in ARMS.items():
        rows = [get(i, pq[i]) for i in sel]
        w, l, p = wl(list(zip(rows, base)), "def3"); fw, fl, fp = wl(list(zip(rows, base)), "hit3")
        print(f"   {name:<36} Chunk W/L {w:>3}/{l:<3} p={p:.2e}   File W/L {fw:>3}/{fl:<3} p={fp:.2e}"
              f"   {'STRICT (0 losses)' if l == 0 else f'{l} chunk loss(es)'}")


if args.set:
    new = [i for i, r in enumerate(pq) if r["id"].startswith("N")]
    old = [i for i, r in enumerate(pq) if r["id"].startswith("S")]
    report("PRIMARY: new queries only (confirmatory, unseen in development)", new)
    report("All queries (frozen 150 + new)", list(range(len(pq))))
    report("Frozen 150 only (development set, for reference)", old)
else:
    report("Frozen 150", list(range(len(pq))))
print("=" * 96)
print(f" Held-out alphas per fold: {res['cv']['alphas']}")
