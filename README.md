# Miru Hybrid Search Benchmark: The Top-K Illusion

This repository contains an isolated, reproducible benchmark demonstrating a critical candidate-depth anomaly in hybrid code retrieval.

By running this benchmark against the Miru Code Search Engine, we empirically demonstrate that deep-pool fetching (`topK=50`) actively degrades search accuracy, because a deeper candidate pool lets more generically named files into a file-path boost that outweighs fused relevance (see [Root cause](#root-cause-bm25-depth--path-stem-boost)). Restricting the native query to `topK=3` leverages Miru's internal $5 \times K$ fetch multiplier as an early-exit relevance filter, boosting Hit@3 by **7.3 points (76.0% → 83.3%)** with zero regressions and ~33% lower median latency.

> [!IMPORTANT]
> **Embedder status:** all results below were produced with Miru running on a **local MiniLM-L6 embedder (256-d)** as a stand-in for Takara's production **DS1** embedder, pending a DS1 API key. Miru's search pipeline (BM25, dense retrieval, RRF fusion, boosts, reranking) is unchanged; only the embedding model differs. Absolute Hit@3 and the size of the gap are expected to differ under DS1. The benchmark switches to DS1 automatically when `TAKARA_API_KEY` is set, and DS1 results will be added here once available.

## Quick Start

```bash
# Install dependencies
npm install

# Run the comparative benchmark
npm run benchmark
```

`npm run benchmark` expects a Miru index of the pinned corpus at `.miru/client-go/index`. On a fresh clone, build it first (see [Reproducing from scratch](#reproducing-from-scratch)).

## Results

`kubernetes/client-go@2761958` · 13,471 chunks · 150 queries · 5 interleaved repetitions per arm · embedder: **local MiniLM-L6-256 (DS1 stand-in)**

| Configuration | Request | Hit@3 | Correct | Δ vs Legacy | Wins / Losses vs Legacy | p50 latency | p95 latency |
|---|---|---:|---:|---:|---:|---:|---:|
| Legacy | `topK=50`, sliced to top 3 | 76.0% | 114 / 150 | — | — | 31.5 ms | 39.7 ms |
| Mid-bound | `topK=5`, sliced to top 3 | 79.3% | 119 / 150 | +3.3 pts | 5 / 0 (p = 0.063) | 21.6 ms | 30.6 ms |
| **Optimized** | `topK=3` | **83.3%** | **125 / 150** | **+7.3 pts** | **11 / 0 (p < 0.001)** | **21.2 ms** | **30.6 ms** |

- **Accuracy rises as K falls.** The Optimized arm fixes 11 queries that Legacy misses and breaks none of the queries Legacy gets right.
- **Faster.** Median search time drops by 33%. It doesn't drop by 94% because Miru scores the whole corpus (BM25 over every chunk, dense similarity over the full index) on every query, whatever `topK` is. Only candidate selection, fusion, boosting and reranking get cheaper.
- The machine-readable output is written to `benchmark/results.<embedder>.json`.

## How It Works

For each query, Miru's `hybridSearch` (`@takara-ai/miru-code` v1.12.2, `src/search.ts`) does the following:

1. **Fetch.** It scores every chunk with BM25 and with dense cosine similarity, then keeps the top `topK × MIRU_RANK_CANDIDATE_MULTIPLIER` from **each** list. The multiplier defaults to **5**; location-style queries use a separate multiplier of 10.
2. **Fuse.** Each list is converted to rank scores, `1 / (10 + rank)` (Reciprocal Rank Fusion, k = 10). They're combined as `α · dense + (1 − α) · BM25`, where `α` = 0.25 for natural-language queries and 0.30 for symbol-like queries. A chunk missing from a list gets 0 from that list.
3. **Boost and rerank.** For natural-language queries it adds file-path *stem boosts* to every pool member. A partial boost of `β · maxScore · (fraction of query words found in the file name or parent directory)` is added, plus a further `β · maxScore` if the file name equals a query word. `β` = `MIRU_RANK_EXACT_STEM_BOOST`, default 0.75. It then applies path penalties and per-file saturation and returns the top `topK`.

| Request | Candidates per list | Unique candidates entering fusion |
|---|---:|---:|
| `topK=3` | 15 | ≤ 30 |
| `topK=5` | 25 | ≤ 50 |
| `topK=50` | 250 | ≤ 500 |

Fused scores are small and tightly packed (at most 1/11 ≈ 0.091), while a single exact-stem match adds `0.75 × maxScore`. A boosted file therefore only has to be *in* the pool to compete, and pool depth decides which boosted files are there. With `topK=3`, a chunk must rank in the top 15 of at least one list to be considered at all. The multiplier therefore acts as a relevance filter before boosting.

**Control: depth, not slicing.** Keep `topK=3` but raise the multiplier to 50, Miru's maximum (`npm run benchmark -- --multiplier 50`). Each list then fetches 150 candidates, and the Optimized arm loses its advantage completely:

| Run | Candidates per list | Hit@3 |
|---|---:|---:|
| `topK=50`, default multiplier (Legacy) | 250 | 76.0% (114 / 150) |
| `topK=3`, multiplier 50 | 150 | 76.0% (114 / 150) |
| `topK=3`, default multiplier (Optimized) | 15 | 83.3% (125 / 150) |

146 of the 150 top-3 lists from `topK=3 × 50` are identical to Legacy's. In the same `--multiplier 50` run, all three arms score 76.0% (114 / 150). Once each list is at least 150 deep, requesting fewer results no longer helps. So the accuracy depends on how deep Miru fetches **before** fusion, not on how many results the caller keeps. The default `npm run benchmark` unsets this variable so that it measures Miru's shipped default; the control output is in `benchmark/results.minilm-l6-256.mult50.json`.

### Root cause: BM25 depth × path-stem boost

`npm run ablation` runs a split-depth copy of Miru's `hybridSearch`, built from Miru's own exported ranking functions. It is verified **output-identical to `MiruIndex.search` on 150/150 queries at topK 3, 5 and 50**. With it, the dense and BM25 depths, the final cut-off, the fusion weight and each post-fusion stage can be varied independently. Results are written to `benchmark/results.ablation.minilm-l6-256.json`.

| Experiment (depth 15 vs 250 per list) | Depth 15 | Depth 250 | Depth penalty | W/L |
|---|---:|---:|---:|---:|
| Full Miru pipeline | 83.3% | 76.0% | **+7.3** (p < 0.001) | 11/0 |
| Final cut-off 3 vs 50 at fixed depth | — | — | **0.0** | 0/0 |
| Deepen **BM25 only** (dense fixed at 75) | 83.3% | 76.0% | **+7.3** (p < 0.001) | 11/0 |
| Deepen **dense only** (BM25 fixed at 75) | 78.0% | 78.0% | 0.0 (p = 1.0) | 1/1 |
| Path-stem boosts off (`β = 0`) | 86.0% | 86.0% | **0.0** | 0/0 |
| No boosts, no rerank (fusion only) | 85.3% | 84.7% | +0.7 (n.s.) | 1/0 |
| `β = 0.25 / 0.5 / 0.75 / 1.5` | 87.3 / 87.3 / 83.3 / 77.3% | 87.3 / 85.3 / 76.0 / 50.0% | 0.0 / +2.0 / +7.3 / **+27.3** | — |

The full dense × BM25 grid and all confidence intervals are in the paper.

- **It's depth, not truncation.** The final cut-off changes nothing.
- **It's not semantic noise.** Deepening the dense list alone has no effect, and the penalty persists even when dense scores carry all the fusion weight (`α = 1`).
- **The boost is necessary.** Without it, depth doesn't matter. With it, the penalty grows monotonically with its strength.
- **The culprits.** On the 11 queries Optimized fixes, all 33 results that displaced the answer carried the stem boost, and 28 are files whose name equals a query word: `interface.go`, `role.go`, `eviction.go`, `scale.go`, `event.go`, `endpoint.go`, `pod.go`. Their median boost is 2.8× their own fused score. Depth helps them in two ways: 15 are admitted only by deep lists, and 15 gain a second list's RRF credit.

The full write-up is in [paper/topk_anomaly.tex](paper/topk_anomaly.tex).

### Recommendation

Request the number of results you actually need. Fetching a large pool and slicing it client-side costs both accuracy and latency. For engine designers: make path boosts rank-aware, for example by applying them only within the top-N fused candidates or scaling them by each candidate's own score, or decouple candidate depth from `topK`. A weaker `β` (0.25–0.5) scored 87.3% here, but it was not tuned on held-out data, so treat it as a hypothesis.

## Methodology

- **Corpus:** [`kubernetes/client-go`](https://github.com/kubernetes/client-go) pinned to commit `2761958e61dbfe479992e21e1320ca462f556b99`.
- **Query set:** [`benchmark/queries.topk.json`](benchmark/queries.topk.json) holds 150 frozen queries. Each query is a Go identifier split into lowercase words (for example, `LeaseCandidateInterface` becomes `lease candidate interface`) and lists every file in the corpus that defines that identifier (`definedIn`, always at least two files). The test suite checks that every listed file really defines its identifier at the pinned commit.
- **Metric:** **Hit@3** counts a query as a hit when any `definedIn` file appears among the top 3 results returned.
- **Arms:** all three arms query the same index with the same query vectors. Only the requested `topK` changes.
- **Repetitions:** each query gets one warm-up call, then runs 5 times per arm, with the arms interleaved. Arm order rotates from query to query so that no arm gets a systematic cache or warm-up advantage. Latency is the per-query median over repetitions, summarised as p50 and p95 across queries.
- **Significance:** an exact two-sided sign test on paired per-query outcomes (wins vs. losses against Legacy).
- **Latency scope:** Miru search time only. With the local embedder, query vectors are precomputed and injected. With DS1, latency includes Miru's own query-embedding call.

## Reproducing from Scratch

```bash
npm install
npm run setup:corpus   # clone client-go and check out the pinned commit
npm run index          # build the Miru index at .miru/client-go/index
npm run benchmark      # run the three arms (options: --reps <n>, --index <path>, --multiplier <n>)
npm run ablation       # root-cause ablation, about 4 min (option: --reuse to re-score saved output)
npm test               # metric, query-set and corpus integrity checks
```

**Embedder selection:**

| Environment | Embedder | Notes |
|---|---|---|
| `TAKARA_API_KEY` set | Takara DS1 (`ds1-miru-int8`) | Production configuration |
| unset (default) | Local MiniLM-L6, 256-d (`@xenova/transformers`) | Offline; fully reproducible without credentials |

You can force either embedder with `EMBEDDER=ds1` or `EMBEDDER=minilm`. The index and the benchmark must use the same embedder, so re-run `npm run index` if you switch. Indexing with the local embedder is CPU-bound and can take 1–2 hours.

## Repository Layout

```
benchmark/
  topk_eval.ts          # benchmark driver: embeds queries, runs arms, prints table
  depth_ablation.ts     # root-cause ablation: six experiments + per-query attribution
  metrics.ts            # Hit@3, MRR@3, percentiles, sign test, Wilson CI, paired bootstrap
  queries.topk.json     # frozen 150-query test set
  results.*.json        # saved outputs behind every number in this README and the paper
scripts/
  miru_topk.ts          # Bun worker that runs MiruIndex.search for each arm
  miru_depth_ablation.ts  # Bun worker: fidelity-checked split-depth copy of Miru's hybridSearch
  index_repo.ts         # builds the Miru index for the pinned corpus
  miru_index_runner.ts  # Bun-side indexing helper
src/
  embedder.ts           # DS1 / local MiniLM embedder selection
  minilm.ts             # local MiniLM-L6 embedder (256-d)
  local-embedding-server.ts  # OpenAI-compatible endpoint for local indexing
paper/
  topk_anomaly.tex      # paper (LaTeX source; build with pdflatex or Overleaf)
tests/
  run.ts                # test suite
```

## Caveats

- **Embedder.** The published numbers use the local MiniLM-256 embedder. The candidate-depth mechanism does not depend on the embedder, but absolute Hit@3 under DS1 will differ, so run with `TAKARA_API_KEY` to get production figures.
- **Single corpus.** Results come from one Go repository. Corpora of other sizes and languages may show different effect sizes.
- **Query style.** The queries are identifier phrases that target named definitions. Free-form conceptual or multi-file queries were not measured.

## License

The benchmark code and query set in this repository are released under the [MIT License](LICENSE).

This repository does not include or redistribute the corpus: `npm run setup:corpus` clones it from upstream, and [`kubernetes/client-go`](https://github.com/kubernetes/client-go) is licensed under Apache-2.0. [`@takara-ai/miru-code`](https://www.npmjs.com/package/@takara-ai/miru-code) and the other dependencies remain under their own licences.
