/**
 * Run under Bun (Miru is Bun-only). Dumps Miru's unmodified Top-K candidate pool per query, with chunk
 * content, so an external (Python) reranker can be evaluated post-retrieval without touching Miru.
 *
 *   bun scripts/miru_candidates.ts <indexDir> <queries.json> <out.json> <topK> [queryVectors.json]
 *
 * Latency is the median of 3 timed searches after one warm-up (Miru search time only when vectors are
 * injected from the local embedder).
 */
import { MiruIndex } from "@takara-ai/miru-code";
import { readFileSync, writeFileSync } from "node:fs";

const [indexDir, queriesPath, outPath, topKArg, vecPath] = process.argv.slice(2);
if (!indexDir || !queriesPath || !outPath) {
  throw new Error("usage: miru_candidates.ts <indexDir> <queries.json> <out.json> <topK> [queryVectors.json]");
}
const topK = Math.max(3, Number(topKArg ?? 25));
const queries = (JSON.parse(readFileSync(queriesPath, "utf-8")) as { queries: { id: string; query: string }[] }).queries;
const vectors: Record<string, number[]> = vecPath ? JSON.parse(readFileSync(vecPath, "utf-8")) : {};
const index = await MiruIndex.loadFromDisk(indexDir, process.env.MIRU_OPENAI_EMBEDDING_MODEL);

const out = [];
for (const q of queries) {
  const queryVector = vectors[q.id] ? Float32Array.from(vectors[q.id]) : undefined;
  await index.search({ query: q.query, queryVector, topK }); // warm-up
  const ms: number[] = [];
  let results: Awaited<ReturnType<typeof index.search>> = [];
  for (let r = 0; r < 3; r++) {
    const t0 = performance.now();
    results = await index.search({ query: q.query, queryVector, topK });
    ms.push(performance.now() - t0);
  }
  ms.sort((a, b) => a - b);
  out.push({
    id: q.id,
    miruMs: ms[1],
    candidates: results.map((x, rank) => ({
      rank,
      score: x.score,
      file_path: x.chunk.file_path,
      start_line: x.chunk.start_line,
      end_line: x.chunk.end_line,
      language: x.chunk.language,
      content: x.chunk.content,
    })),
  });
}
writeFileSync(outPath, JSON.stringify({ runtime: `bun ${Bun.version}`, chunks: index.chunks.length, topK, queries: out }));
console.log(`wrote ${out.length} candidate pools (topK=${topK}) -> ${outPath}`);
