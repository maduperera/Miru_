/**
 * Miru Top-K benchmark: does requesting fewer results from Miru change which 3 results come back?
 *
 *   npm run benchmark                 # = tsx benchmark/topk_eval.ts [--reps 5] [--index .miru/client-go/index]
 *
 * Arms (Hit@3 = a file that defines the queried identifier is among the top 3):
 *   Legacy     topK=50, sliced to the first 3
 *   Mid-bound  topK=5,  sliced to the first 3
 *   Optimized  topK=3
 *
 * Query embeddings are computed once here (local MiniLM) and injected, so the latency columns are Miru search
 * time only. With TAKARA_API_KEY (DS1) Miru embeds each query itself and latency includes that API call.
 * Writes benchmark/results.<embedder>.json.
 */
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createEmbedder, embedderChoice } from "../src/embedder.ts";
import { hitAt3, paired, percentile } from "./metrics.ts";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const flag = (name: string, fallback: string) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const REPS = Number(flag("--reps", "5"));
const INDEX = path.resolve(ROOT, flag("--index", ".miru/client-go/index"));
const WORK = path.join(ROOT, ".bench", "topk");

interface Query { id: string; query: string; identifier: string; definedIn: string[] }
const spec = JSON.parse(readFileSync(path.join(ROOT, "benchmark", "queries.topk.json"), "utf-8")) as { repo: string; commit: string; queries: Query[] };
if (!existsSync(INDEX)) {
  console.error(`No Miru index at ${path.relative(ROOT, INDEX)}. Run: npm run setup:corpus && npm run index`);
  process.exit(1);
}

// 1. Query vectors (local embedder) — DS1 lets Miru embed in-process instead.
mkdirSync(WORK, { recursive: true });
const useLocal = embedderChoice() !== "ds1";
const emb = createEmbedder();
const vecPath = path.join(WORK, `queryvecs.${emb.name}.json`);
if (useLocal && !existsSync(vecPath)) {
  process.stdout.write(`Embedding ${spec.queries.length} queries with ${emb.name}… `);
  const vecs: Record<string, number[]> = {};
  for (const q of spec.queries) vecs[q.id] = Array.from(await emb.embedQuery(q.query));
  writeFileSync(vecPath, JSON.stringify(vecs));
  console.log("done");
}
const queriesPath = path.join(WORK, "queries.json");
writeFileSync(queriesPath, JSON.stringify({ queries: spec.queries.map(({ id, query }) => ({ id, query })) }));

// 2. Run the three arms against Miru under Bun.
const rawPath = path.join(WORK, `raw.${emb.name}.json`);
console.log(`Running ${spec.queries.length} queries × 3 arms × ${REPS} reps against Miru…`);
await new Promise<void>((resolve, reject) => {
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    ...(useLocal ? { TAKARA_API_KEY: "local-embedder-no-key", MIRU_OPENAI_EMBEDDING_MODEL: emb.name, MIRU_EMBEDDING_DIMENSIONS: String(emb.dim) } : {}),
    MIRU_FLOAT_VECTORS: "1", MIRU_NO_UPDATE_CHECK: "1", MIRU_QUIET: "1",
    MIRU_CACHE_HOME: path.join(ROOT, ".miru", "_cache"), MIRU_CREDENTIALS_DIR: path.join(ROOT, ".miru", "_creds"),
  };
  delete env.MIRU_RANK_CANDIDATE_MULTIPLIER; // benchmark Miru's shipped default (5)
  const child = spawn(path.join(ROOT, "node_modules", ".bin", "bun"),
    ["scripts/miru_topk.ts", INDEX, queriesPath, rawPath, String(REPS), ...(useLocal ? [vecPath] : [])],
    { cwd: ROOT, env, stdio: ["ignore", "inherit", "inherit"] });
  child.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`bun exited with ${code}`))));
});
const raw = JSON.parse(readFileSync(rawPath, "utf-8")) as { runtime: string; chunks: number; queries: { id: string; top3: Record<string, string[]>; ms: Record<string, number[]> }[] };

// 3. Score.
const byId = new Map(spec.queries.map((q) => [q.id, q]));
const ARMS = [
  { k: 50, name: "Legacy", detail: "topK=50 → first 3" },
  { k: 5, name: "Mid-bound", detail: "topK=5 → first 3" },
  { k: 3, name: "Optimized", detail: "topK=3" },
];
const n = raw.queries.length;
const rows = ARMS.map((a) => {
  const hitVec = raw.queries.map((q) => hitAt3(q.top3[a.k], byId.get(q.id)!.definedIn));
  const perQueryMs = raw.queries.map((q) => percentile(q.ms[a.k], 50)); // median over reps, per query
  return { ...a, hitVec, hits: hitVec.filter(Boolean).length, p50: percentile(perQueryMs, 50), p95: percentile(perQueryMs, 95) };
});
const legacy = rows[0];

// 4. Terminal table.
const table = (head: string[], body: string[][], align: ("l" | "r")[]) => {
  const w = head.map((h, i) => Math.max(h.length, ...body.map((r) => r[i].length)));
  const cell = (s: string, i: number) => (align[i] === "r" ? s.padStart(w[i]) : s.padEnd(w[i]));
  const line = (l: string, m: string, r: string) => l + w.map((x) => "─".repeat(x + 2)).join(m) + r;
  return [line("┌", "┬", "┐"), "│ " + head.map(cell).join(" │ ") + " │", line("├", "┼", "┤"), ...body.map((r) => "│ " + r.map(cell).join(" │ ") + " │"), line("└", "┴", "┘")].join("\n");
};
const body = rows.map((r) => {
  const d = (100 * (r.hits - legacy.hits)) / n;
  const pr = paired(r.hitVec, legacy.hitVec);
  return [
    `${r.name} (${r.detail})`,
    `${((100 * r.hits) / n).toFixed(1)}%`,
    `${r.hits}/${n}`,
    r === legacy ? "—" : `${d >= 0 ? "+" : ""}${d.toFixed(1)} pts`,
    r === legacy ? "—" : `${pr.wins}/${pr.losses}, ${pr.p < 0.001 ? "p<0.001" : `p=${pr.p.toFixed(3)}`}`,
    r.p50.toFixed(1),
    r.p95.toFixed(1),
  ];
});
console.log(`\nMiru Top-K benchmark · ${spec.repo}@${spec.commit.slice(0, 7)} (${raw.chunks.toLocaleString("en-US")} chunks) · ${n} queries · embedder ${emb.name} · ${raw.runtime} · ${REPS} reps\n`);
console.log(table(["Configuration", "Hit@3", "Correct", "Δ vs Legacy", "W/L vs Legacy (sign test)", "p50 ms", "p95 ms"], body, ["l", "r", "r", "r", "r", "r", "r"]));
console.log(`\nHit@3: a file defining the queried identifier is in the top 3. Latency: Miru search time per query (median of ${REPS} reps), ${useLocal ? "query embedding excluded (precomputed)" : "including the DS1 query-embedding call"}.`);

const outFile = path.join(ROOT, "benchmark", `results.${emb.name}.json`);
writeFileSync(outFile, JSON.stringify({
  repo: spec.repo, commit: spec.commit, chunks: raw.chunks, queries: n, embedder: emb.name, runtime: raw.runtime, reps: REPS,
  arms: rows.map((r) => ({ name: r.name, topK: r.k, hitAt3: r.hits / n, hits: r.hits, p50Ms: r.p50, p95Ms: r.p95, vsLegacy: r === legacy ? null : paired(r.hitVec, legacy.hitVec) })),
  perQuery: raw.queries.map((q, i) => ({ id: q.id, ...Object.fromEntries(rows.map((r) => [`hit_topK${r.k}`, r.hitVec[i]])) })),
}, null, 2));
console.log(`Results → ${path.relative(ROOT, outFile)}`);
