/**
 * Produces Miru Top-K candidate pools for the Kalpana fusion benchmark (no change to Miru's pipeline).
 *
 *   npx tsx benchmark/dump_candidates.ts [--k 25] [--index .miru/client-go/index]
 *
 * Same environment as benchmark/topk_eval.ts: query vectors from the local embedder (cached in
 * .bench/topk/queryvecs.<embedder>.json), Miru's shipped candidate multiplier, float vectors.
 * Writes .bench/kalpana/candidates.<embedder>.k<K>.json
 */
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createEmbedder, embedderChoice } from "../src/embedder.ts";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const flag = (name: string, fallback: string) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const K = Number(flag("--k", "25"));
const INDEX = path.resolve(ROOT, flag("--index", ".miru/client-go/index"));
const QFILE = flag("--queries", "benchmark/queries.topk.json");
// "queries.topk.json" -> "" (legacy names), "queries.topk.all500.json" -> ".all500"
const QSET = path.basename(QFILE).replace(/^queries\.topk\.?/, "").replace(/\.?json$/, "");
const SET_TAG = QSET ? `.${QSET}` : "";
const VEC_DIR = path.join(ROOT, ".bench", "topk");
const WORK = path.join(ROOT, ".bench", "kalpana");

const spec = JSON.parse(readFileSync(path.resolve(ROOT, QFILE), "utf-8")) as {
  queries: { id: string; query: string }[];
};
if (!existsSync(INDEX)) {
  console.error(`No Miru index at ${path.relative(ROOT, INDEX)}. Run: npm run setup:corpus && npm run index`);
  process.exit(1);
}
mkdirSync(VEC_DIR, { recursive: true });
mkdirSync(WORK, { recursive: true });

const useLocal = embedderChoice() !== "ds1";
const emb = createEmbedder();
const vecPath = path.join(VEC_DIR, `queryvecs.${emb.name}.json`);
if (useLocal) {
  const vecs: Record<string, number[]> = existsSync(vecPath) ? JSON.parse(readFileSync(vecPath, "utf-8")) : {};
  const missing = spec.queries.filter((q) => !vecs[q.id]);
  if (missing.length) {
    process.stdout.write(`Embedding ${missing.length} new queries with ${emb.name}… `);
    for (const q of missing) vecs[q.id] = Array.from(await emb.embedQuery(q.query));
    writeFileSync(vecPath, JSON.stringify(vecs));
    console.log("done");
  }
}
const queriesPath = path.join(WORK, `queries${SET_TAG}.json`);
writeFileSync(queriesPath, JSON.stringify({ queries: spec.queries.map(({ id, query }) => ({ id, query })) }));
const outPath = path.join(WORK, `candidates.${emb.name}${SET_TAG}.k${K}.json`);

await new Promise<void>((resolve, reject) => {
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    ...(useLocal
      ? { TAKARA_API_KEY: "local-embedder-no-key", MIRU_OPENAI_EMBEDDING_MODEL: emb.name, MIRU_EMBEDDING_DIMENSIONS: String(emb.dim) }
      : {}),
    MIRU_FLOAT_VECTORS: "1", MIRU_NO_UPDATE_CHECK: "1", MIRU_QUIET: "1",
    MIRU_CACHE_HOME: path.join(ROOT, ".miru", "_cache"), MIRU_CREDENTIALS_DIR: path.join(ROOT, ".miru", "_creds"),
  };
  delete env.MIRU_RANK_CANDIDATE_MULTIPLIER; // Miru's shipped default
  const child = spawn(path.join(ROOT, "node_modules", ".bin", "bun"),
    ["scripts/miru_candidates.ts", INDEX, queriesPath, outPath, String(K), ...(useLocal ? [vecPath] : [])],
    { cwd: ROOT, env, stdio: ["ignore", "inherit", "inherit"] });
  child.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`bun exited with ${code}`))));
});
