/**
 * Run under Bun. Depth-ablation worker for Miru's hybrid search.
 *
 *   bun scripts/miru_depth_ablation.ts <indexDir> <queries.json> <queryVectors.json> <out.json>
 *
 * `splitHybridSearch` below is Miru 1.12.2's `hybridSearch` (src/search.ts) line for line, built from Miru's own
 * exported ranking functions, with one change: the dense and BM25 candidate depths are separate parameters
 * instead of a single `topK × MIRU_RANK_CANDIDATE_MULTIPLIER`. A fidelity check against `MiruIndex.search` runs
 * first; it is reported alongside the results so readers can confirm the reimplementation is exact.
 *
 * Every configuration is deterministic (precomputed query vectors), so each runs once per query.
 */
import { MiruIndex } from "@takara-ai/miru-code";
import { readFileSync, writeFileSync } from "node:fs";
import { applyQueryBoost, boostMultiChunkFiles } from "../node_modules/@takara-ai/miru-code/src/ranking/boosting.ts";
import { rankingConfig } from "../node_modules/@takara-ai/miru-code/src/ranking/config.ts";
import { searchImprovementsEnabled } from "../node_modules/@takara-ai/miru-code/src/ranking/features.ts";
import { isLocationQuery } from "../node_modules/@takara-ai/miru-code/src/ranking/location.ts";
import { rerankTopk } from "../node_modules/@takara-ai/miru-code/src/ranking/penalties.ts";
import { resolveAlpha } from "../node_modules/@takara-ai/miru-code/src/ranking/weighting.ts";
import { selectorToMask } from "../node_modules/@takara-ai/miru-code/src/index/sparse.ts";
import { selectTopKScoreIndices } from "../node_modules/@takara-ai/miru-code/src/index/topk.ts";
import { tokenize } from "../node_modules/@takara-ai/miru-code/src/tokens.ts";
import { chunkKey } from "../node_modules/@takara-ai/miru-code/src/types.ts";
import type { Chunk } from "../node_modules/@takara-ai/miru-code/src/types.ts";

const [indexDir, queriesPath, vecPath, outPath] = process.argv.slice(2);
if (!indexDir || !queriesPath || !vecPath || !outPath) {
  throw new Error("usage: miru_depth_ablation.ts <indexDir> <queries.json> <queryVectors.json> <out.json>");
}
const queries = (JSON.parse(readFileSync(queriesPath, "utf-8")) as { queries: { id: string; query: string }[] }).queries;
const vectors: Record<string, number[]> = JSON.parse(readFileSync(vecPath, "utf-8"));
const index = await MiruIndex.loadFromDisk(indexDir, process.env.MIRU_OPENAI_EMBEDDING_MODEL);
// biome-ignore lint: Miru keeps these as TS-private fields; they are plain properties at runtime.
const internals = index as unknown as { bm25Index: any; semanticIndex: any; content?: string[] };
const chunks: Chunk[] = index.chunks;
const chunksByKey = new Map(chunks.map((c) => [chunkKey(c), c]));
const rerankDefault = internals.content?.includes("code") ?? true; // same default as MiruIndex.search

/** Per-candidate provenance, recorded for traced configurations. */
interface Trace { file: string; line: number; denseRank: number | null; bm25Rank: number | null; fused: number; boosted: number }

interface SplitOptions { denseDepth: number; bm25Depth: number; topK: number; alpha?: number | null; rerank?: boolean; trace?: boolean }

function rrfScores(scores: Map<string, number>, rrfK: number): Map<string, number> {
  if (scores.size === 0) return scores;
  const ranked = [...scores.entries()].sort((a, b) => b[1] - a[1]);
  const out = new Map<string, number>();
  ranked.forEach(([key], i) => out.set(key, 1.0 / (rrfK + i + 1)));
  return out;
}

async function splitHybridSearch(query: string, queryVec: Float32Array, o: SplitOptions) {
  const alphaWeight = resolveAlpha(query, o.alpha);
  const tuning = rankingConfig();

  // BM25 path (Miru's searchBm25, depth = o.bm25Depth)
  const tokens = tokenize(query);
  const bm25Hits: { key: string; score: number }[] = [];
  if (tokens.length) {
    const scores: number[] = await internals.bm25Index.getScoresAsync(tokens, selectorToMask(undefined, chunks.length));
    for (const i of selectTopKScoreIndices(scores, o.bm25Depth)) {
      const s = scores[i];
      if (s !== undefined && chunks[i] !== undefined && s > 0) bm25Hits.push({ key: chunkKey(chunks[i]), score: s });
    }
  }
  // Dense path (Miru's semanticFromQueryVector, depth = o.denseDepth)
  const { indices, distances } = internals.semanticIndex.query(queryVec, o.denseDepth, undefined);
  const semantic: { key: string; score: number }[] = [];
  (indices as number[]).forEach((idx, i) => {
    if (chunks[idx]) semantic.push({ key: chunkKey(chunks[idx]), score: 1.0 - (distances[i] ?? 0) });
  });

  const semanticScores = new Map(semantic.map((r) => [r.key, r.score] as const));
  const bm25Scores = new Map<string, number>();
  for (const r of bm25Hits) if (r.score) bm25Scores.set(r.key, r.score);
  const normalizedSemantic = rrfScores(semanticScores, tuning.rrfK);
  const normalizedBm25 = rrfScores(bm25Scores, tuning.rrfK);

  const allKeys = new Set([...normalizedSemantic.keys(), ...normalizedBm25.keys()]);
  const sortedKeys = [...allKeys].sort((a, b) => (chunksByKey.get(a) as Chunk).start_line - (chunksByKey.get(b) as Chunk).start_line);
  const combinedScores = new Map<string, number>();
  for (const key of sortedKeys) {
    combinedScores.set(key, alphaWeight * (normalizedSemantic.get(key) ?? 0) + (1 - alphaWeight) * (normalizedBm25.get(key) ?? 0));
  }
  const fused = o.trace ? new Map(combinedScores) : null;

  let ranked: [Chunk, number][];
  let boosted: Map<string, number> | null = null;
  const rerank = o.rerank ?? rerankDefault;
  if (rerank) {
    boostMultiChunkFiles(combinedScores, chunksByKey);
    applyQueryBoost(combinedScores, query, chunks, chunksByKey);
    if (o.trace) boosted = new Map(combinedScores);
    ranked = rerankTopk(combinedScores, chunksByKey, o.topK, alphaWeight < 1.0);
  } else {
    if (o.trace) boosted = new Map(combinedScores);
    ranked = [...combinedScores.entries()].sort((a, b) => b[1] - a[1]).slice(0, o.topK).map(([k, s]) => [chunksByKey.get(k) as Chunk, s]);
  }

  const result = { top: ranked.map(([c]) => ({ file: c.file_path, line: c.start_line })), pool: allKeys.size } as {
    top: { file: string; line: number }[]; pool: number; trace?: Trace[];
  };
  if (o.trace && fused && boosted) {
    const denseRank = new Map([...normalizedSemantic.keys()].map((k, i) => [k, i + 1] as const));
    const bm25Rank = new Map([...normalizedBm25.keys()].map((k, i) => [k, i + 1] as const));
    // Full provenance for every pool member (the analysis picks out top-3 intruders and target chunks).
    result.trace = [...allKeys].map((k) => {
      const c = chunksByKey.get(k) as Chunk;
      return { file: c.file_path, line: c.start_line, denseRank: denseRank.get(k) ?? null, bm25Rank: bm25Rank.get(k) ?? null, fused: fused.get(k) ?? 0, boosted: boosted.get(k) ?? 0 };
    });
  }
  return result;
}

// ---------------------------------------------------------------------------------------------------------------
// Experiment plan
// ---------------------------------------------------------------------------------------------------------------
interface Config { id: string; group: string; denseDepth: number; bm25Depth: number; topK: number; alpha?: number | null; rerank?: boolean; env?: Record<string, string>; trace?: boolean }
const configs: Config[] = [];
const add = (c: Config) => configs.push(c);

// E1 symmetric depth sweep (final topK = 3)
for (const d of [3, 5, 10, 15, 25, 50, 75, 100, 150, 250, 500]) add({ id: `sym_d${d}`, group: "sweep", denseDepth: d, bm25Depth: d, topK: 3 });
// E2 asymmetric dense × BM25 depth grid (final topK = 3)
const G = [15, 30, 75, 150, 250];
for (const D of G) for (const B of G) add({ id: `grid_D${D}_B${B}`, group: "grid", denseDepth: D, bm25Depth: B, topK: 3 });
// E3 2×2 factorial: candidate depth × final rerank topK (Legacy = d250/k50, Optimized = d15/k3)
for (const d of [15, 250]) for (const k of [3, 50]) add({ id: `fact_d${d}_k${k}`, group: "factorial", denseDepth: d, bm25Depth: d, topK: k, trace: true });
// E4 fusion weight α (0 = BM25 only, null = Miru default, 1 = dense only)
for (const a of [0, null, 0.5, 1]) for (const d of [15, 250]) add({ id: `alpha_${a ?? "default"}_d${d}`, group: "alpha", denseDepth: d, bm25Depth: d, topK: 3, alpha: a });
// E5 post-fusion stage ablations at both depths
for (const d of [15, 250]) {
  add({ id: `stage_full_d${d}`, group: "stage", denseDepth: d, bm25Depth: d, topK: 3 });
  add({ id: `stage_nostem_d${d}`, group: "stage", denseDepth: d, bm25Depth: d, topK: 3, env: { MIRU_RANK_EXACT_STEM_BOOST: "0" } });
  add({ id: `stage_nopenalty_d${d}`, group: "stage", denseDepth: d, bm25Depth: d, topK: 3, env: { MIRU_RANK_PATH_PENALTIES: "0" } });
  add({ id: `stage_fusiononly_d${d}`, group: "stage", denseDepth: d, bm25Depth: d, topK: 3, rerank: false });
}
// E6 dose-response: path-stem boost strength β (Miru default 0.75) at both depths
for (const b of ["0", "0.1", "0.25", "0.5", "0.75", "1.5"]) for (const d of [15, 250]) {
  add({ id: `beta_${b}_d${d}`, group: "beta", denseDepth: d, bm25Depth: d, topK: 3, env: { MIRU_RANK_EXACT_STEM_BOOST: b } });
}

// ---------------------------------------------------------------------------------------------------------------
// Fidelity check: splitHybridSearch(depth = K × multiplier) must equal MiruIndex.search(topK = K) exactly.
// ---------------------------------------------------------------------------------------------------------------
const tuning = rankingConfig();
const fidelity: Record<string, { identical: number; total: number; mismatches: string[] }> = {};
for (const K of [3, 5, 50]) {
  const f = (fidelity[`topK${K}`] = { identical: 0, total: 0, mismatches: [] as string[] });
  for (const q of queries) {
    const vec = Float32Array.from(vectors[q.id]);
    const depth = searchImprovementsEnabled() && isLocationQuery(q.query) ? K * tuning.locationCandidateMultiplier : K * tuning.candidateMultiplier;
    const ref = (await index.search({ query: q.query, queryVector: vec, topK: K })).map((r) => `${r.chunk.file_path}:${r.chunk.start_line}`);
    const mine = (await splitHybridSearch(q.query, vec, { denseDepth: depth, bm25Depth: depth, topK: K })).top.map((r) => `${r.file}:${r.line}`);
    f.total++;
    if (ref.join("|") === mine.join("|")) f.identical++;
    else f.mismatches.push(q.id);
  }
  console.log(`fidelity topK=${K}: ${f.identical}/${f.total} identical to MiruIndex.search`);
}
const locationQueries = queries.filter((q) => searchImprovementsEnabled() && isLocationQuery(q.query)).map((q) => q.id);

// ---------------------------------------------------------------------------------------------------------------
// Run all configurations.
// ---------------------------------------------------------------------------------------------------------------
const results: Record<string, { id: string; top: { file: string; line: number }[]; pool: number; trace?: Trace[] }[]> = {};
const t0 = performance.now();
for (const [ci, c] of configs.entries()) {
  const saved: Record<string, string | undefined> = {};
  for (const [k, v] of Object.entries(c.env ?? {})) { saved[k] = process.env[k]; process.env[k] = v; }
  const rows = [];
  for (const q of queries) {
    const r = await splitHybridSearch(q.query, Float32Array.from(vectors[q.id]), c);
    rows.push({ id: q.id, ...r });
  }
  results[c.id] = rows;
  for (const [k, v] of Object.entries(saved)) { if (v === undefined) delete process.env[k]; else process.env[k] = v; }
  if ((ci + 1) % 10 === 0 || ci === configs.length - 1) console.log(`  ${ci + 1}/${configs.length} configurations (${((performance.now() - t0) / 1000).toFixed(0)} s)`);
}

writeFileSync(outPath, JSON.stringify({
  runtime: `bun ${Bun.version}`, chunks: chunks.length,
  miru: { rrfK: tuning.rrfK, candidateMultiplier: tuning.candidateMultiplier, exactStemBoost: tuning.exactStemBoost, definitionBoost: tuning.definitionBoost, fileSaturationDecay: tuning.fileSaturationDecay },
  fidelity, locationQueries, configs, results,
}));
