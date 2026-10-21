#!/bin/sh
# Per-repo multi-language pipeline: Miru topK=3/5 pools -> tree-sitter roles -> Kalpana benchmark -> paired tests.
#   sh scripts/run_lang.sh <repo>
set -e
R=$1
Q=benchmark/queries.$R.json
B=.bench/kalpana/candidates.minilm-l6-256.queries.$R
for K in 3 5; do
  npx tsx benchmark/dump_candidates.ts --k $K --queries $Q --index .miru/$R/index
  npx tsx scripts/extract_roles.ts pool $B.k$K.json $B.k$K.ts.json
done
python3 miru_kalpana/benchmark_kalpana_fusion.py --extractor treesitter --examples 0 \
  --candidates $B.k5.ts.json --queries $Q
python3 miru_kalpana/rerank_vs_native.py --repo $R
