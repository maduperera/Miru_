#!/bin/sh
# Full Miru-suite run (Jordan's benchmarks, 9 languages):
#   CodeSearchNet x6 (go java javascript php python ruby; 500 each)
#   + Miru curated A/B repos (typescript python rust go javascript elixir ruby; file-level labels)
#   sh scripts/run_miru_suite.sh            (needs network for the first two steps)
set -e
node --import tsx tests/roles_smoke.ts                 # sanity: every language yields roles
node --import tsx scripts/prep_codesearchnet.ts
# Miru's curated query list lives in its source repo (not in the npm package); pin the commit we used.
if [ ! -d eval-repos/miru-code ]; then
  git clone --quiet https://github.com/takara-ai/miru-code eval-repos/miru-code
  git -C eval-repos/miru-code checkout --quiet e4fac7f5140475def202049c523037b10fcb3504
fi
node --import tsx scripts/prep_miru_ab.ts
mkdir -p benchmark/logs
for R in csn-go csn-java csn-javascript csn-php csn-python csn-ruby \
         ab-miru-code ab-flask ab-fastapi ab-click ab-axum ab-serde ab-gin ab-cobra ab-express ab-phoenix ab-rails; do
  [ -f benchmark/queries.$R.json ] || { echo "skip $R"; continue; }
  npx tsx scripts/index_repo.ts $R
  sh scripts/run_lang.sh $R 2>&1 | tee benchmark/logs/$R.log
done
