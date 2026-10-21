#!/usr/bin/env python3
"""
benchmark_kalpana_fusion.py
---------------------------
Non-breaking benchmark: Kalpana-RIF as an additive, post-retrieval reranker on top of Miru's
unmodified Top-K candidate pool (kubernetes/client-go, 150 ambiguous-symbol queries).

    python3 miru_kalpana/benchmark_kalpana_fusion.py [--k 25] [--candidates PATH] [--examples 5]
                                                     [--rebuild-tensors] [--folds 5]

Pipeline (Miru is never modified):
  1. Miru Top-K pools come from benchmark/dump_candidates.ts -> scripts/miru_candidates.ts (Bun).
     Generated automatically if the candidates file is missing.
  2. OFFLINE (simulated index time): every unique chunk in the pools -> Go (role, token) pairs ->
     Kalpana HRR tensor, serialized once to .bench/kalpana/tensors.<tag>.npz.
  3. ONLINE (per query): query -> structural tensor bundle{ROLE_FUNC (x) q, ROLE_CLASS (x) q}
     ("where is <q> *defined*"), then one mat-vec against the pre-computed pool tensors.
  4. Arms:
       Baseline         Miru's own order
       Kalpana          pure structural reorder
       RRF              reciprocal-rank fusion of Miru rank + Kalpana rank (rank-only reference)
       Linear(alpha)    (1-alpha)*minmax(miru_score) + alpha*minmax(kalpana_cos), per query
     Alpha is swept over a grid. Because choosing alpha on the evaluation queries is optimistic,
     a K-fold cross-validated arm (alpha picked on train folds, scored on held-out) is reported too.

Reranking permutes the pool, so pool Recall@K is identical across arms by construction.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from miru_kalpana_reranker import KalpanaReranker  # noqa: E402

ASTPair = Tuple[str, str]
RRF_C = 60
MAX_PAIRS = 96  # cap per chunk: bounds latency and bundle crosstalk
ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
CV_SEED = 0x5E5510

# ---------------------------------------------------------------------------
# Go -> (role, token) pairs  (regex fallback; no tree-sitter dependency)
# ---------------------------------------------------------------------------
_GO_KEYWORDS = frozenset(
    "if for switch select return func go defer range make new len cap append panic recover "
    "copy delete close map chan struct interface type var const package import else case "
    "default break continue goto fallthrough nil true false string int bool error byte rune".split())

_RE_FUNC = re.compile(r"^\s*func\s+(?:\(\s*(\w+)?\s*\*?\s*[\w\.\[\], ]+\)\s*)?(\w+)\s*(?:\[[^\]]*\])?\s*\(([^)]*)\)", re.M)
_RE_TYPE = re.compile(r"^\s*type\s+(\w+)\b", re.M)
_RE_TYPE_BLOCK = re.compile(r"^\s*type\s*\(\s*$", re.M)
_RE_BLOCK_TYPE = re.compile(r"^\t(\w+)\s+(?:struct|interface|func|map|\[\]|\*|[A-Za-z_][\w\.]*)", re.M)
_RE_CALL = re.compile(r"(?<!func )\b((?:\w+\.)*?)(\w+)\s*\(")
_RE_VAR = re.compile(r"\b(\w+)\s*(?:,\s*\w+\s*)*:=|\bvar\s+(\w+)\b")
_RE_PARAM = re.compile(r"(?:^|,)\s*(\w+)(?:\s*,\s*\w+)*\s+[\*\[\]\w\.\.\.]")
_RE_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_RE_STRING = re.compile(r'"(?:\\.|[^"\\])*"|`[^`]*`')
_RE_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+")


def split_ident(name: str) -> str:
    """ResourceSliceIndexers -> 'resource slice indexers' (matches the query phrasing)."""
    return " ".join(w.lower() for w in _RE_CAMEL.findall(name))


def go_pairs(content: str) -> List[ASTPair]:
    src = _RE_STRING.sub('""', _RE_COMMENT.sub("", content))
    pairs: List[ASTPair] = []
    defined = set()

    for m in _RE_FUNC.finditer(src):
        name = m.group(2)
        defined.add(name)
        pairs.append(("ROLE_FUNC", name))
        for p in _RE_PARAM.finditer(m.group(3) or ""):
            pairs.append(("ROLE_ARG", p.group(1)))
    for m in _RE_TYPE.finditer(src):
        defined.add(m.group(1))
        pairs.append(("ROLE_CLASS", m.group(1)))
    if _RE_TYPE_BLOCK.search(src):
        for m in _RE_BLOCK_TYPE.finditer(src):
            defined.add(m.group(1))
            pairs.append(("ROLE_CLASS", m.group(1)))
    for m in _RE_VAR.finditer(src):
        name = m.group(1) or m.group(2)
        if name and name != "_" and name not in _GO_KEYWORDS:
            pairs.append(("ROLE_VAR", name))
    for m in _RE_CALL.finditer(src):
        name = m.group(2)
        if name not in _GO_KEYWORDS and name not in defined:
            pairs.append(("ROLE_CALL", name))

    seen, out = set(), []
    for role, ident in pairs:  # definitions first (above), dedupe, then cap
        tok = split_ident(ident)
        if tok and (role, tok) not in seen:
            seen.add((role, tok))
            out.append((role, tok))
    return out[:MAX_PAIRS]


_SKIP_ARGS = {"self", "cls", "_"}


def treesitter_pairs(c: dict) -> List[ASTPair]:
    """Roles precomputed by scripts/extract_roles.ts; same normalisation as go_pairs."""
    seen, out = set(), []
    for role, ident in c.get("roles") or []:
        if role == "ROLE_ARG" and ident in _SKIP_ARGS:
            continue
        tok = split_ident(ident)
        if tok and (role, tok) not in seen:
            seen.add((role, tok))
            out.append((role, tok))
    return out[:MAX_PAIRS]


def query_pairs(query: str) -> List[ASTPair]:
    q = query.strip().lower()
    return [("ROLE_FUNC", q), ("ROLE_CLASS", q)]


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
_GENERIC_BASENAMES = frozenset({
    "interface.go", "generated_expansion.go", "doc.go", "register.go", "types.go",
    "expansion_generated.go", "factory.go", "generic.go",
})


def is_generic(path: str) -> bool:
    p = path.replace("\\", "/")
    base = p.rsplit("/", 1)[-1]
    return (base in _GENERIC_BASENAMES
            or base.startswith("fake_") or "/fake/" in p
            or base.endswith("_test.go")                                  # test suites
            or (base.startswith("zz_generated.") and base.endswith(".go")))  # generated bindings


def in_defined(path: str, defined_in: Sequence[str]) -> bool:
    p = path.replace("\\", "/")
    return any(p == d or p.endswith("/" + d) for d in defined_in)


def is_distractor(path: str, defined_in: Sequence[str]) -> bool:
    """Generic pattern AND not ground truth: e.g. expansion_generated.go is the *answer* for
    '... lister expansion' queries and must not be counted as a distractor there."""
    return is_generic(path) and not in_defined(path, defined_in)


def defines(content: str, identifier: str) -> bool:
    ident = re.escape(identifier)
    return bool(re.search(rf"^\s*func\s+(?:\([^)]*\)\s*)?{ident}\b|^\s*type\s+{ident}\b|^\t{ident}\s+(?:struct|interface)\b",
                          content, re.M))


def chunk_key(c: dict) -> str:
    return f'{c["file_path"]}:{c["start_line"]}:{c["end_line"]}'


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def arm_metrics(order: Sequence[int], cands: List[dict], q: dict) -> dict:
    file_hit = [in_defined(cands[i]["file_path"], q["definedIn"]) for i in order]
    def _defines(c: dict) -> bool:
        if "defs" in c and c.get("language") != "go":  # tree-sitter ground truth for non-Go pools
            return q["identifier"] in c["defs"]
        return defines(c["content"], q["identifier"])
    def_hit = [file_hit[j] and _defines(cands[i]) for j, i in enumerate(order)]
    first = next((r for r, h in enumerate(file_hit) if h), None)
    first_def = next((r for r, h in enumerate(def_hit) if h), None)
    top3 = list(order[:3])
    return {
        "hit1": bool(file_hit) and file_hit[0],
        "hit3": any(file_hit[:3]),
        "def3": any(def_hit[:3]),
        "p3": sum(file_hit[:3]) / 3,
        "rr": 0.0 if first is None else 1.0 / (first + 1),
        "rr_def": 0.0 if first_def is None else 1.0 / (first_def + 1),
        "generic3": sum(is_distractor(cands[i]["file_path"], q["definedIn"]) for i in top3),
        "generic1": bool(top3) and is_distractor(cands[top3[0]]["file_path"], q["definedIn"]),
        "top3": [f'{cands[i]["file_path"]}:{cands[i]["start_line"]}' for i in top3],
        "top3_hit": file_hit[:3],
    }


def aggregate(rows: List[dict], base_rows: Optional[List[dict]] = None) -> dict:
    n = len(rows)
    a = {
        "hit1": sum(r["hit1"] for r in rows) / n,
        "hit3": sum(r["hit3"] for r in rows) / n,
        "def3": sum(r["def3"] for r in rows) / n,
        "p3": statistics.mean(r["p3"] for r in rows),
        "mrr": statistics.mean(r["rr"] for r in rows),
        "mrr_def": statistics.mean(r["rr_def"] for r in rows),
        "generic3": statistics.mean(r["generic3"] for r in rows) / 3,
        "generic1": sum(r["generic1"] for r in rows) / n,
    }
    if base_rows is not None:
        g1 = [(b, r) for b, r in zip(base_rows, rows) if b["generic1"]]
        a["suppress1"] = (sum(not r["generic1"] for _, r in g1) / len(g1)) if g1 else 0.0
        a["baseGeneric1N"] = len(g1)
        for key in ("hit3", "def3"):
            w = sum(r[key] and not b[key] for b, r in zip(base_rows, rows))
            l = sum(b[key] and not r[key] for b, r in zip(base_rows, rows))
            a[f"{key}_wins"], a[f"{key}_losses"], a[f"{key}_p"] = w, l, sign_test(w, l)
    return a


def select_key(a: dict) -> Tuple[float, float]:
    """Alpha selection objective: Chunk-in-Top-3 first, MRR (file) as tie-break."""
    return (round(a["def3"], 9), round(a["mrr"], 9))


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1)))))
    return xs[i]


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def linear_order(miru: np.ndarray, kal: np.ndarray, alpha: float) -> List[int]:
    s = (1 - alpha) * minmax(miru) + alpha * minmax(kal)
    return list(np.argsort(-s, kind="stable"))  # ties keep Miru's order


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _legacy(paths):
    # candidates.<embedder>.k<K>.json only (query-set-tagged dumps have an extra dotted segment)
    return sorted(p for p in paths if p.name.count(".") == 3)


def load_candidates(k: int, path: Optional[str]) -> Tuple[dict, Path]:
    if path:
        p = Path(path)
    else:
        matches = _legacy((ROOT / ".bench" / "kalpana").glob(f"candidates.*.k{k}.json"))
        p = matches[0] if matches else None
        if p is None:
            print(f"No candidate pools for K={k}; generating via Miru (unchanged pipeline)…")
            subprocess.run(["npx", "tsx", "benchmark/dump_candidates.ts", "--k", str(k)], cwd=ROOT, check=True)
            matches = _legacy((ROOT / ".bench" / "kalpana").glob(f"candidates.*.k{k}.json"))
            if not matches:
                sys.exit("candidate dump did not produce a file")
            p = matches[0]
    return json.loads(p.read_text()), p


def build_tensor_cache(rr: KalpanaReranker, data: dict, path: Path, rebuild: bool,
                       extractor: str = "regex") -> Tuple[Dict[str, int], np.ndarray, np.ndarray, dict]:
    """OFFLINE: tensorize every unique pool chunk once and serialize (simulated index time)."""
    if path.exists() and not rebuild:
        z = np.load(path, allow_pickle=False)
        keys = [str(k) for k in z["keys"]]
        info = {"source": "cache", "chunks": len(keys), "buildS": float(z["build_s"])}
        return {k: i for i, k in enumerate(keys)}, z["mat"], z["valid"], info

    uniq: Dict[str, dict] = {}
    for row in data["queries"]:
        for c in row["candidates"]:
            uniq.setdefault(chunk_key(c), c)
    keys = list(uniq)
    mat = np.zeros((len(keys), rr.vocab.dim), dtype=np.float64)
    valid = np.zeros(len(keys), dtype=bool)
    print(f"Offline: tensorizing {len(keys)} unique pool chunks…", flush=True)
    t0 = time.perf_counter()
    for i, k in enumerate(keys):
        pairs = treesitter_pairs(uniq[k]) if extractor == "treesitter" else go_pairs(uniq[k]["content"])
        if pairs:
            mat[i] = rr.index_chunk(pairs)
            valid[i] = True
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(keys)}", flush=True)
    build_s = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, keys=np.array(keys), mat=mat, valid=valid, build_s=np.array(build_s))
    return {k: i for i, k in enumerate(keys)}, mat, valid, {"source": "built", "chunks": len(keys), "buildS": build_s}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=25, help="Miru candidate pool size (default 25)")
    ap.add_argument("--candidates", help="explicit candidates JSON (skips Miru)")
    ap.add_argument("--examples", type=int, default=5, help="changed-ranking examples to print")
    ap.add_argument("--rebuild-tensors", action="store_true", help="ignore the serialized tensor cache")
    ap.add_argument("--folds", type=int, default=5, help="cross-validation folds for alpha selection")
    ap.add_argument("--queries", default="benchmark/queries.topk.json", help="query spec (must cover the pools)")
    ap.add_argument("--extractor", choices=["regex", "treesitter"], default="regex",
                    help="chunk role source: Go regex, or 'roles' from scripts/extract_roles.ts")
    args = ap.parse_args()

    spec = json.loads((ROOT / args.queries).read_text())
    qby = {q["id"]: q for q in spec["queries"]}
    data, cpath = load_candidates(args.k, args.candidates)
    tag = cpath.stem.replace("candidates.", "")
    if args.extractor == "treesitter" and not tag.endswith(".ts"):
        tag += ".ts"  # never share a cache/results file with the regex extractor

    print("Loading Kalpana-RIF (MiniLM codebook)…", flush=True)
    t0 = time.perf_counter()
    rr = KalpanaReranker()
    load_s = time.perf_counter() - t0

    # ---- OFFLINE: pre-computed pool tensors (index time) -------------------
    kidx, mat, valid, cache_info = build_tensor_cache(
        rr, data, ROOT / ".bench" / "kalpana" / f"tensors.{tag}.npz", args.rebuild_tensors, args.extractor)
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0] = 1.0
    unit = mat / norms[:, None]  # row-normalized once, offline

    # ---- ONLINE: per-query query tensor + mat-vec --------------------------
    arm_names = ["baseline", "kalpana", "rrf"] + [f"lin{a}" for a in ALPHAS]
    per_query, miru_ms, lat_qcold, lat_qwarm, lat_score = [], [], [], [], []
    for row in data["queries"]:
        q, cands = qby[row["id"]], row["candidates"]
        if not cands:
            continue
        miru_ms.append(row.get("miruMs", 0.0))
        rows_idx = np.array([kidx[chunk_key(c)] for c in cands])

        # Cold = phrase NOT in the token cache. Query phrases usually equal split identifier names
        # that the offline pass already embedded, so evict them to measure a genuine MiniLM encode.
        rr.vocab._tokens.pop(q["query"].strip().lower(), None)
        t_a = time.perf_counter()
        q_t = rr.index_chunk(query_pairs(q["query"]))  # cold: query phrase not yet embedded
        t_b = time.perf_counter()
        qn = q_t / (np.linalg.norm(q_t) or 1.0)
        kal = unit[rows_idx] @ qn
        kal_order = list(np.argsort(-kal, kind="stable"))
        t_c = time.perf_counter()
        rr.index_chunk(query_pairs(q["query"]))  # warm: phrase now in the token cache
        t_d = time.perf_counter()
        lat_qcold.append((t_b - t_a) * 1e3)
        lat_score.append((t_c - t_b) * 1e3)
        lat_qwarm.append((t_d - t_c) * 1e3)

        miru = np.array([c["score"] for c in cands], dtype=np.float64)
        base = list(range(len(cands)))
        krank = {i: r for r, i in enumerate(kal_order)}
        orders = {
            "baseline": base,
            "kalpana": kal_order,
            "rrf": sorted(base, key=lambda i: -(1 / (RRF_C + i + 1) + 1 / (RRF_C + krank[i] + 1))),
            **{f"lin{a}": linear_order(miru, kal, a) for a in ALPHAS},
        }
        per_query.append({
            "id": q["id"], "query": q["query"], "identifier": q["identifier"],
            "poolRecall": any(in_defined(c["file_path"], q["definedIn"]) for c in cands),
            "parsed": float(valid[rows_idx].mean()),
            **{a: arm_metrics(orders[a], cands, q) for a in arm_names},
        })

    n = len(per_query)
    base_rows = [r["baseline"] for r in per_query]
    agg = {a: aggregate([r[a] for r in per_query], base_rows) for a in arm_names}
    pool_recall = sum(r["poolRecall"] for r in per_query) / n

    best_alpha = max(ALPHAS, key=lambda a: select_key(agg[f"lin{a}"]))
    best_def3 = max(ALPHAS, key=lambda a: (agg[f"lin{a}"]["def3"], -abs(a - best_alpha)))
    best_mrr = max(ALPHAS, key=lambda a: (agg[f"lin{a}"]["mrr"], -abs(a - best_alpha)))

    # ---- Cross-validated alpha (honest held-out estimate) ------------------
    ids = list(range(n))
    random.Random(CV_SEED).shuffle(ids)
    folds = [ids[f::args.folds] for f in range(args.folds)]
    cv_rows: List[Optional[dict]] = [None] * n
    cv_alphas = []
    for f, test in enumerate(folds):
        train = [i for i in ids if i not in set(test)]
        tb = [per_query[i]["baseline"] for i in train]
        a_f = max(ALPHAS, key=lambda a: select_key(aggregate([per_query[i][f"lin{a}"] for i in train], tb)))
        cv_alphas.append(a_f)
        for i in test:
            cv_rows[i] = per_query[i][f"lin{a_f}"]
    agg["cv"] = aggregate(cv_rows, base_rows)  # type: ignore[arg-type]

    # ----------------------------------------------------------------- report
    W = 92
    K = data.get("topK", args.k)
    print("\n" + "=" * W)
    print(f" Miru + Kalpana-RIF fusion benchmark v2   ({data.get('runtime', '?')}, {data.get('chunks', '?')} chunks)")
    print(f" {n} queries | Miru pool K={K} | pool Recall@K = {pool_recall:.1%} (identical for all arms)")
    print(" Distractor = generic pattern AND not in the query's ground-truth file list")
    print("=" * W)
    cols = [("baseline", "Baseline"), ("kalpana", "Kalpana"), ("rrf", "RRF"),
            (f"lin{best_alpha}", f"Lin a={best_alpha}*"), ("cv", f"Lin CV{args.folds}†")]
    print(f" {'Metric':<34}" + "".join(f"{h:>11}" for _, h in cols))
    print(" " + "-" * (W - 1))

    def line(label, key, fmt="{:.1%}", lower_better=False):
        b = agg["baseline"][key]
        cells = []
        for j, (a, _) in enumerate(cols):
            v = agg[a][key]
            s = fmt.format(v)
            if j and abs(v - b) > 1e-9:
                s += " ▲" if ((v < b) if lower_better else (v > b)) else " ▼"
            else:
                s += "  "
            cells.append(f"{s:>11}")
        print(f" {label:<34}" + "".join(cells))

    line("Top-1  (defining file at rank 1)", "hit1")
    line("Hit@3  (defining file in top 3)", "hit3")
    line("Def@3  (defining chunk in top 3)", "def3")
    line("Precision@3", "p3")
    line("MRR@K  (file)", "mrr", "{:.3f}")
    line("MRR@K  (defining chunk)", "mrr_def", "{:.3f}")
    line("Distractor share of top 3", "generic3", lower_better=True)
    line("Distractor at rank 1", "generic1", lower_better=True)
    line(f"Rank-1 distractor suppressed n={agg['baseline']['baseGeneric1N']}", "suppress1")
    print(" " + "-" * (W - 1))
    print(f" * alpha chosen on all {n} queries (in-sample, optimistic).")
    print(f" † alpha chosen on train folds, scored on held-out folds; per-fold alpha = {cv_alphas}")

    print("\n Alpha sweep   final = (1-a)*minmax(miru_score) + a*minmax(kalpana_cos)")
    print(" " + "-" * (W - 1))
    print(f" {'alpha':>6}{'Top-1':>9}{'Hit@3':>9}{'Def@3':>9}{'MRR':>8}{'MRRdef':>8}"
          f"{'Hit@3 W/L':>12}{'p':>8}{'Def@3 W/L':>12}{'p':>8}")
    for a in ALPHAS:
        g = agg[f"lin{a}"]
        mark = " *" if a == best_alpha else ""
        print(f" {a:>6}{g['hit1']:>9.1%}{g['hit3']:>9.1%}{g['def3']:>9.1%}{g['mrr']:>8.3f}{g['mrr_def']:>8.3f}"
              f"{g['hit3_wins']:>6}/{g['hit3_losses']:<5}{g['hit3_p']:>8.4f}{g['def3_wins']:>6}/{g['def3_losses']:<5}{g['def3_p']:>8.4f}{mark}")
    print(" " + "-" * (W - 1))
    print(f" argmax Def@3 -> a={best_def3}   argmax MRR -> a={best_mrr}   selected (Def@3, tie MRR) -> a={best_alpha}")
    for a, label in ((f"lin{best_alpha}", f"Lin a={best_alpha} (in-sample)"), ("cv", f"Lin CV{args.folds} (held-out)"), ("rrf", "RRF")):
        g = agg[a]
        print(f"   {label:<26} Hit@3 W/L {g['hit3_wins']}/{g['hit3_losses']} p={g['hit3_p']:.4f}   "
              f"Def@3 W/L {g['def3_wins']}/{g['def3_losses']} p={g['def3_p']:.4f}")

    print("\n Latency / query (ms) — pool tensors pre-computed offline")
    print(" " + "-" * (W - 1))
    print(f" {'':<44}{'p50':>9}{'p95':>9}")
    print(f"   {'Miru search (Top-' + str(K) + ')':<42}{pct(miru_ms, 50):>9.3f}{pct(miru_ms, 95):>9.3f}")
    print(f"   {'Kalpana query tensor, cold (MiniLM encode)':<42}{pct(lat_qcold, 50):>9.3f}{pct(lat_qcold, 95):>9.3f}")
    print(f"   {'Kalpana query tensor, warm (token cached)':<42}{pct(lat_qwarm, 50):>9.3f}{pct(lat_qwarm, 95):>9.3f}")
    print(f"   {'Kalpana pool scoring (mat-vec + sort)':<42}{pct(lat_score, 50):>9.3f}{pct(lat_score, 95):>9.3f}")
    cold = [a + b for a, b in zip(lat_qcold, lat_score)]
    warm = [a + b for a, b in zip(lat_qwarm, lat_score)]
    print(f"   {'Overhead above Miru, cold query':<42}{pct(cold, 50):>9.3f}{pct(cold, 95):>9.3f}"
          f"   {'< 1 ms' if pct(cold, 95) < 1 else '>= 1 ms'} at p95")
    print(f"   {'Overhead above Miru, warm query':<42}{pct(warm, 50):>9.3f}{pct(warm, 95):>9.3f}"
          f"   {'< 1 ms' if pct(warm, 95) < 1 else '>= 1 ms'} at p95")
    print(f" Offline tensor build: {cache_info['chunks']} chunks in {cache_info['buildS']:.1f}s "
          f"({cache_info['source']}); model load {load_s:.1f}s (one-time)")
    print(f" Chunks with ≥1 extracted pair: {statistics.mean(r['parsed'] for r in per_query):.1%}")
    print("=" * W)

    best = f"lin{best_alpha}"
    changed = [r for r in per_query if r[best]["def3"] != r["baseline"]["def3"]][: args.examples]
    if changed:
        print(f"\n Side-by-side Top-3 where Def@3 flipped under Lin a={best_alpha} (first {len(changed)}):")
        for r in changed:
            tag_ = "WIN " if r[best]["def3"] else "LOSS"
            print(f"\n [{tag_}] {r['id']}  \"{r['query']}\"  -> {r['identifier']}")
            b, k = r["baseline"], r[best]
            for j in range(3):
                bs = (("✔ " if b["top3_hit"][j] else "  ") + b["top3"][j]) if j < len(b["top3"]) else ""
                ks = (("✔ " if k["top3_hit"][j] else "  ") + k["top3"][j]) if j < len(k["top3"]) else ""
                print(f"   {j + 1}. {bs[:42]:<43} | {ks[:42]}")

    out = ROOT / "benchmark" / f"results.kalpana.{tag}.json"
    out.write_text(json.dumps({
        "candidates": str(cpath.relative_to(ROOT)) if cpath.is_relative_to(ROOT) else str(cpath),
        "queries": n, "poolRecallAtK": pool_recall, "rrfC": RRF_C, "alphas": ALPHAS,
        "bestAlphaInSample": best_alpha, "argmaxDef3": best_def3, "argmaxMRR": best_mrr,
        "cv": {"folds": args.folds, "seed": CV_SEED, "alphas": cv_alphas},
        "arms": agg,
        "latencyMs": {
            "miru": {"p50": pct(miru_ms, 50), "p95": pct(miru_ms, 95)},
            "queryTensorCold": {"p50": pct(lat_qcold, 50), "p95": pct(lat_qcold, 95)},
            "queryTensorWarm": {"p50": pct(lat_qwarm, 50), "p95": pct(lat_qwarm, 95)},
            "poolScoring": {"p50": pct(lat_score, 50), "p95": pct(lat_score, 95)},
            "overheadCold": {"p50": pct(cold, 50), "p95": pct(cold, 95)},
            "overheadWarm": {"p50": pct(warm, 50), "p95": pct(warm, 95)},
            "offlineBuildS": cache_info["buildS"], "modelLoadS": load_s,
        },
        "perQuery": per_query,
    }, indent=1))
    print(f"\n Results -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
