#!/usr/bin/env python3
"""
gen_queries.py
--------------
Re-implementation of the sampling rule documented in benchmark/queries.topk.json, used to draw
NEW ambiguous-symbol queries from kubernetes/client-go. The frozen 150-query file is not modified.

    python3 miru_kalpana/gen_queries.py --n 350 [--seed 0x5e5511]

Rule (from queries.topk.json "description"):
  * exported Go identifier, func (incl. methods) or type
  * >= 8 chars, >= 2 words
  * defined in 2..30 different directories
  * test, testdata, openapi and applyconfigurations paths excluded
  * sampled with a fixed seed
Word split and the "defines" regex are copied from tests/run.ts so the new queries pass the same
checks. Identifiers already in the frozen set are excluded.

Writes:
  benchmark/queries.topk.new350.json   (the new queries only; ids N0..)
  benchmark/queries.topk.all500.json   (frozen 150 + new 350)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "eval-repos" / "client-go"

# Candidate definitions: top-level funcs, methods, and types (single or grouped `type (...)`).
_RE_DEF = re.compile(r"^(?:func\s+(?:\([^)]*\)\s*)?|type\s+|\t)([A-Z]\w*)\b", re.M)


def words(ident: str) -> str:
    """Identical to tests/run.ts words()."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", ident)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return s.lower()


def defines(src: str, ident: str) -> bool:
    """Identical to the tests/run.ts corpus check."""
    return re.search(rf"(?:func\s+(?:\([^)]*\)\s*)?|type\s+){re.escape(ident)}\b", src) is not None


def excluded(rel: str) -> bool:
    parts = rel.split("/")
    return (rel.endswith("_test.go")
            or any("test" in p for p in parts[:-1])          # test/, testing/, testdata/ dirs
            or any(p in ("openapi", "applyconfigurations") for p in parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=350)
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=0x5E5511)
    args = ap.parse_args()

    frozen = json.loads((ROOT / "benchmark" / "queries.topk.json").read_text())
    taken = {q["identifier"] for q in frozen["queries"]}

    files: dict[str, str] = {}
    for dp, _, fns in os.walk(CORPUS):
        if "/.git" in dp:
            continue
        for fn in fns:
            if fn.endswith(".go"):
                rel = os.path.relpath(os.path.join(dp, fn), CORPUS).replace(os.sep, "/")
                if not excluded(rel):
                    files[rel] = (Path(dp) / fn).read_text(errors="ignore")

    cand: dict[str, set] = defaultdict(set)
    for rel, src in files.items():
        for m in _RE_DEF.finditer(src):
            cand[m.group(1)].add(rel)

    pool = []
    for ident, fs in cand.items():
        if ident in taken or len(ident) < 8 or len(words(ident).split()) < 2:
            continue
        fs = sorted(f for f in fs if defines(files[f], ident))  # same check as tests/run.ts
        dirs = {os.path.dirname(f) for f in fs}
        if len(fs) >= 2 and 2 <= len(dirs) <= 30:
            pool.append((ident, fs))
    pool.sort()
    print(f"eligible identifiers (excluding frozen 150): {len(pool)}")
    if len(pool) < args.n:
        raise SystemExit(f"only {len(pool)} eligible identifiers; cannot draw {args.n}")

    rng = random.Random(args.seed)
    picked = rng.sample(pool, args.n)
    new = [{"id": f"N{i}", "query": words(ident), "identifier": ident, "definedIn": fs}
           for i, (ident, fs) in enumerate(picked)]

    meta = {k: frozen[k] for k in ("repo", "commit", "metric")}
    desc = (f"{args.n} NEW ambiguous-symbol lookups drawn by miru_kalpana/gen_queries.py with the rule documented "
            f"in queries.topk.json (seed {hex(args.seed)}); frozen identifiers excluded. Fresh confirmatory set.")
    (ROOT / "benchmark" / f"queries.topk.new{args.n}.json").write_text(
        json.dumps({**meta, "description": desc, "queries": new}, indent=1))
    total = len(frozen["queries"]) + len(new)
    (ROOT / "benchmark" / f"queries.topk.all{total}.json").write_text(
        json.dumps({**meta, "description": f"Frozen 150 (S*) + {args.n} new (N*).",
                    "queries": frozen["queries"] + new}, indent=1))
    nd = [len({os.path.dirname(f) for f in q["definedIn"]}) for q in new]
    print(f"wrote {len(new)} new queries; dirs/query min {min(nd)} max {max(nd)}; "
          f"files: new{args.n} + all{total}")


if __name__ == "__main__":
    main()
