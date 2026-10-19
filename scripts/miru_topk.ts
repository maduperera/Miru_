/**
 * Run under Bun (Miru is Bun-only). Times the three Top-K arms on a saved Miru index.
 *
 *   bun scripts/miru_topk.ts <indexDir> <queries.json> <out.json> <reps> [queryVectors.json]
 *
 * Arms: topK=50, topK=5, topK=3 (top-3 files recorded for each). Reps are interleaved and the arm order
 * rotates per query, so no arm systematically benefits from cache warm-up. With queryVectors.json the
 * vectors are injected as `queryVector` (local embedder); without it Miru embeds the query itself (DS1).
 */
import { MiruIndex } from "@takara-ai/miru-code";
import { readFileSync, writeFileSync } from "node:fs";

const [indexDir, queriesPath, outPath, repsArg, vecPath] = process.argv.slice(2);
if (!indexDir || !queriesPath || !outPath) throw new Error("usage: miru_topk.ts <indexDir> <queries.json> <out.json> <reps> [queryVectors.json]");
const reps = Math.max(1, Number(repsArg ?? 5));
const ARMS = [50, 5, 3] as const;

const queries = (JSON.parse(readFileSync(queriesPath, "utf-8")) as { queries: { id: string; query: string }[] }).queries;
const vectors: Record<string, number[]> = vecPath ? JSON.parse(readFileSync(vecPath, "utf-8")) : {};
const index = await MiruIndex.loadFromDisk(indexDir, process.env.MIRU_OPENAI_EMBEDDING_MODEL);

const out = queries.map((q, qi) => ({ id: q.id, top3: {} as Record<number, string[]>, ms: {} as Record<number, number[]>, qi }));
for (const row of out) {
  const q = queries[row.qi];
  const queryVector = vectors[q.id] ? Float32Array.from(vectors[q.id]) : undefined;
  await index.search({ query: q.query, queryVector, topK: 3 }); // warm-up
  const order = ARMS.map((_, i) => ARMS[(i + row.qi) % ARMS.length]);
  for (let r = 0; r < reps; r++) {
    for (const k of order) {
      const t0 = performance.now();
      const results = await index.search({ query: q.query, queryVector, topK: k });
      (row.ms[k] ??= []).push(performance.now() - t0);
      row.top3[k] = results.slice(0, 3).map((x) => x.chunk.file_path);
    }
  }
}
writeFileSync(outPath, JSON.stringify({ runtime: `bun ${Bun.version}`, chunks: index.chunks.length, reps, arms: ARMS, queries: out.map(({ qi, ...r }) => r) }));
