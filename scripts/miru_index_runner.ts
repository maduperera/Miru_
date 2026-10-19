/**
 * Run under Bun (Miru is Bun-only). Builds a Miru index with the unmodified @takara-ai/miru-code
 * and writes the bundle to <outDir>.
 *   bun scripts/miru_index_runner.ts <repoPath> <outDir>
 * Embeddings come from whatever MIRU_OPENAI_BASE_URL points at (the local MiniLM server when
 * launched via scripts/index_repo.ts without a TAKARA_API_KEY, otherwise Takara DS1).
 */
import { MiruIndex } from "@takara-ai/miru-code";

const [repoPath, outDir] = process.argv.slice(2);
if (!repoPath || !outDir) throw new Error("usage: miru_index_runner.ts <repoPath> <outDir>");

const t0 = performance.now();
const index = await MiruIndex.fromPath(repoPath);
const buildMs = performance.now() - t0;
await index.save(outDir);
console.log(JSON.stringify({ repoPath, outDir, chunks: index.chunks.length, buildMs: Math.round(buildMs) }));
