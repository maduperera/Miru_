/** Plain-assert test runner: `npm test`. */
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { hitAt3, paired, percentile, signTestP } from "../benchmark/metrics.ts";
import { MINILM_DIM, truncateNormalize } from "../src/minilm.ts";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
let passed = 0, skipped = 0;
const test = async (name: string, fn: () => void | Promise<void>) => {
  try { await fn(); passed++; console.log(`  ✓ ${name}`); }
  catch (e) { console.error(`  ✗ ${name}\n${(e as Error).stack}`); process.exitCode = 1; }
};
const skip = (name: string) => { skipped++; console.log(`  - ${name} (skipped)`); };

console.log("metrics");
await test("hitAt3 only looks at the first three results", () => {
  assert.equal(hitAt3(["a.go", "b.go", "c.go", "t.go"], ["t.go"]), false);
  assert.equal(hitAt3(["a.go", "t.go"], ["x.go", "t.go"]), true);
  assert.equal(hitAt3([], ["t.go"]), false);
});
await test("percentile uses nearest rank", () => {
  assert.equal(percentile([5, 1, 3, 2, 4], 50), 3);
  assert.equal(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95), 10);
  assert.ok(Number.isNaN(percentile([], 50)));
});
await test("sign test matches exact binomial values", () => {
  assert.equal(signTestP(0, 0), 1);
  assert.ok(Math.abs(signTestP(11, 0) - 2 / 2 ** 11) < 1e-12); // 0.000977
  assert.ok(Math.abs(signTestP(6, 1) - 0.125) < 1e-12);
  assert.equal(signTestP(3, 3), 1);
});
await test("paired counts wins and losses, ignoring ties", () => {
  assert.deepEqual(paired([true, true, false, false], [false, true, true, false]), { wins: 1, losses: 1, p: 1 });
});

console.log("embedder");
await test("truncateNormalize returns a 256-D unit vector", () => {
  const v = truncateNormalize(Float32Array.from({ length: 384 }, (_, i) => Math.sin(i + 1)));
  assert.equal(v.length, MINILM_DIM);
  assert.ok(Math.abs(Math.hypot(...v) - 1) < 1e-5);
});

console.log("query set (benchmark/queries.topk.json)");
const spec = JSON.parse(readFileSync(path.join(ROOT, "benchmark", "queries.topk.json"), "utf-8")) as { commit: string; queries: { id: string; query: string; identifier: string; definedIn: string[] }[] };
await test("150 unique queries, each with at least two defining files", () => {
  assert.equal(spec.queries.length, 150);
  assert.equal(new Set(spec.queries.map((q) => q.id)).size, 150);
  for (const q of spec.queries) {
    assert.ok(q.query.trim().length > 0 && q.identifier.length >= 8, q.id);
    assert.ok(q.definedIn.length >= 2, `${q.id} should be ambiguous`);
  }
});
await test("query text is the identifier split into lowercase words", () => {
  const words = (id: string) => id.replace(/([a-z0-9])([A-Z])/g, "$1 $2").replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2").toLowerCase();
  for (const q of spec.queries) assert.equal(q.query, words(q.identifier), q.id);
});
const corpus = path.join(ROOT, "eval-repos", "client-go");
if (existsSync(corpus)) {
  await test("every target file exists in the corpus and defines the identifier", () => {
    for (const q of spec.queries) {
      for (const f of q.definedIn) {
        const src = readFileSync(path.join(corpus, f), "utf-8");
        assert.ok(new RegExp(`(?:func\\s+(?:\\([^)]*\\)\\s*)?|type\\s+)${q.identifier}\\b`).test(src), `${q.id}: ${f} does not define ${q.identifier}`);
      }
    }
  });
} else skip("corpus checks (run `npm run setup:corpus`)");

console.log(`\n${passed} passed, ${skipped} skipped${process.exitCode ? " — FAILURES" : ""}`);
