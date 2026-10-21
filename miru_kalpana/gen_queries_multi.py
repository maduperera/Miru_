#!/usr/bin/env python3
"""
gen_queries_multi.py
--------------------
Language-neutral version of gen_queries.py. Draws "find the definition of X" queries from any repo
using the tree-sitter definition map written by scripts/extract_roles.ts (repo mode).

    npx tsx scripts/extract_roles.ts repo eval-repos/<repo> .bench/defs/<repo>.json
    python3 miru_kalpana/gen_queries_multi.py --repo <repo> [--n 300] [--seed 0x5e5512]

Rule (same intent as the Go set):
  * function/method/class/type identifier, >= 8 chars, >= 2 words (camelCase or snake_case)
  * defined in >= 2 files spread over 2..30 directories (ambiguous symbol)
  * test / spec / fixture / example / benchmark / vendored paths excluded
  * sampled with a fixed seed; if fewer than --n are eligible, all are used
Writes benchmark/queries.<repo>.json (ids <REPO>0..).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_EXCL = re.compile(r"(^|/)(tests?|testing|testdata|spec|specs|fixtures?|examples?|benches|benchmarks?|"
                   r"vendor|node_modules|third_party|playground|docs?|__tests__)(/|$)|"
                   r"(_test|_spec|\.test|\.spec)\.\w+$|(^|/)test_[^/]*$|Test[^/]*\.java$")


def words(ident: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", ident)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return " ".join(w for w in re.split(r"[_\s]+", s.lower()) if w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=0x5E5512)
    args = ap.parse_args()

    repo_dir = ROOT / "eval-repos" / args.repo
    dmap = ROOT / ".bench" / "defs" / f"{args.repo}.json"
    if not dmap.exists():
        dmap.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["npx", "tsx", "scripts/extract_roles.ts", "repo", str(repo_dir), str(dmap)],
                       cwd=ROOT, check=True)
    defs: dict = json.loads(dmap.read_text())["defs"]

    pool = []
    for ident, fs in defs.items():
        fs = sorted({f for f in fs if not _EXCL.search(f)})
        dirs = {os.path.dirname(f) for f in fs}
        if len(ident) >= 8 and len(words(ident).split()) >= 2 and len(fs) >= 2 and 2 <= len(dirs) <= 30:
            pool.append((ident, fs))
    pool.sort()
    n = min(args.n, len(pool))
    print(f"{args.repo}: eligible identifiers {len(pool)}; drawing {n}")
    picked = random.Random(args.seed).sample(pool, n)
    tag = re.sub(r"\W", "", args.repo).upper()[:6]
    qs = [{"id": f"{tag}{i}", "query": words(ident), "identifier": ident, "definedIn": fs}
          for i, (ident, fs) in enumerate(picked)]
    commit = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    out = ROOT / "benchmark" / f"queries.{args.repo}.json"
    out.write_text(json.dumps({
        "repo": args.repo, "commit": commit,
        "metric": "defining file / defining chunk in top 3",
        "description": f"{n} ambiguous-symbol lookups drawn by gen_queries_multi.py (seed {hex(args.seed)}) "
                       f"from tree-sitter definitions; >=8 chars, >=2 words, 2..30 dirs, tests excluded.",
        "queries": qs}, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
