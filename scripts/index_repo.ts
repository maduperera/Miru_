/**
 * Builds the Miru index for a corpus under eval-repos/: starts a local OpenAI-compatible embedding endpoint
 * (MiniLM-256) unless a TAKARA_API_KEY selects DS1, then runs the unmodified Miru indexer under Bun.
 *   tsx scripts/index_repo.ts client-go [--int8]
 * Output bundle: .miru/<repo>/index  (bm25_index.json, semantic_index/, chunks.json, metadata.json)
 * Default is float32 vectors (MIRU_FLOAT_VECTORS=1); pass --int8 for Miru's production default storage.
 */
import { spawn } from "node:child_process";
import { mkdirSync, rmSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createEmbedder, embedderChoice } from "../src/embedder.ts";
import { startEmbeddingServer } from "../src/local-embedding-server.ts";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repo = process.argv[2];
const int8 = process.argv.includes("--int8");
if (!repo) throw new Error("usage: index_repo.ts <repo-dir-name under eval-repos/> [--int8]");

const repoPath = path.join(root, "eval-repos", repo);
const outDir = path.join(root, ".miru", repo + (int8 ? "-int8" : ""), "index");
rmSync(outDir, { recursive: true, force: true });
mkdirSync(path.dirname(outDir), { recursive: true });

// DS1 when a key is present (or EMBEDDER=ds1); otherwise a local embedder behind an OpenAI-compatible endpoint.
const embedder = embedderChoice() !== "ds1" ? createEmbedder() : null;
const server = embedder ? await startEmbeddingServer(embedder, 8787) : null;
const bun = path.join(root, "node_modules", ".bin", "bun");
// NOTE: async spawn — spawnSync would block this process's event loop and deadlock the in-process embedding server.
const child = spawn(bun, [path.join(root, "scripts", "miru_index_runner.ts"), repoPath, outDir], {
  cwd: root,
  stdio: "inherit",
  env: {
    ...process.env,
    ...(embedder
      ? {
          TAKARA_API_KEY: "local-embedder-no-key",
          MIRU_OPENAI_BASE_URL: "http://127.0.0.1:8787/v1",
          MIRU_OPENAI_EMBEDDING_MODEL: embedder.name,
          MIRU_EMBEDDING_DIMENSIONS: String(embedder.dim),
          // The local ONNX server is serial and CPU-bound: Miru's default 360-item batches x N workers make each request
          // wait minutes and trip Bun's fetch timeout (TimeoutError). Small batches + 2 workers keep every request short.
          MIRU_EMBEDDING_BATCH_SIZE: "16",
          MIRU_CONCURRENCY: "2",
        }
      : {}),
    MIRU_CACHE_HOME: path.join(root, ".miru", "_cache"),
    MIRU_CREDENTIALS_DIR: path.join(root, ".miru", "_creds"),
    MIRU_NO_UPDATE_CHECK: "1",
    MIRU_QUIET: process.env.MIRU_QUIET ?? "1",
    ...(int8 ? {} : { MIRU_FLOAT_VECTORS: "1" }),
  },
});
const code: number = await new Promise((resolve) => child.on("exit", (c) => resolve(c ?? 1)));
server?.close();
process.exit(code);
