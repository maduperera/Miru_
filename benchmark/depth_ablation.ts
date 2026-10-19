/**
 * Candidate-depth ablation for Miru's hybrid search: which stage of the pipeline makes deep pools hurt?
 *
 *   npm run ablation                  # = tsx benchmark/depth_ablation.ts [--reuse] [--index .miru/client-go/index]
 *
 * Runs scripts/miru_depth_ablation.ts under Bun (a fidelity-checked, split-depth copy of Miru's hybridSearch),
 * then scores five experiments on the 150-query set:
 *   E1 symmetric depth sweep          E2 dense × BM25 depth grid        E3 depth × final-topK factorial
 *   E4 fusion weight α                E5 post-fusion stage ablations    + per-query displacement attribution
 * Writes benchmark/results.ablation.<embedder>.json. --reuse skips the Bun run if raw output already exists.
 */
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createEmbedder, embedderChoice } from "../src/embedder.ts";
import { bootstrapDeltaCI, hitAt3, paired, rrAt3, wilson } from "./metrics.ts";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const flag = (name: string, fallback: string) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const INDEX = path.resolve(ROOT, flag("--index", ".miru/client-go/index"));
const WORK = path.join(ROOT, ".bench", "topk");

interface Query { id: string; query: string; identifier: string; definedIn: string[] }
const spec = JSON.parse(readFileSync(path.join(ROOT, "benchmark", "queries.topk.json"), "utf-8")) as { repo: string; commit: string; queries: Query[] };
const byId = new Map(spec.queries.map((q) => [q.id, q]));
if (embedderChoice() === "ds1") {
  console.error("The ablation injects precomputed query vectors; run it with the local embedder (unset TAKARA_API_KEY) or add DS1 vectors.");
  process.exit(1);
}
const emb = createEmbedder();
const vecPath = path.join(WORK, `queryvecs.${emb.name}.json`);
const rawPath = path.join(WORK, `ablation.${emb.name}.json`);
mkdirSync(WORK, { recursive: true });

// 1. Query vectors + Bun run -------------------------------------------------------------------------------------
if (!existsSync(vecPath)) {
  process.stdout.write(`Embedding ${spec.queries.length} queries with ${emb.name}… `);
  const vecs: Record<string, number[]> = {};
  for (const q of spec.queries) vecs[q.id] = Array.from(await emb.embedQuery(q.query));
  writeFileSync(vecPath, JSON.stringify(vecs));
  console.log("done");
}
const queriesPath = path.join(WORK, "queries.json");
writeFileSync(queriesPath, JSON.stringify({ queries: spec.queries.map(({ id, query }) => ({ id, query })) }));
if (!(args.includes("--reuse") && existsSync(rawPath))) {
  console.log("Running depth ablation under Bun…");
  await new Promise<void>((resolve, reject) => {
    const env: NodeJS.ProcessEnv = {
      ...process.env, TAKARA_API_KEY: "local-embedder-no-key", MIRU_OPENAI_EMBEDDING_MODEL: emb.name, MIRU_EMBEDDING_DIMENSIONS: String(emb.dim),
      MIRU_FLOAT_VECTORS: "1", MIRU_NO_UPDATE_CHECK: "1", MIRU_QUIET: "1",
      MIRU_CACHE_HOME: path.join(ROOT, ".miru", "_cache"), MIRU_CREDENTIALS_DIR: path.join(ROOT, ".miru", "_creds"),
    };
    for (const k of Object.keys(env)) if (k.startsWith("MIRU_RANK_") || k === "MIRU_SEARCH_V2") delete env[k]; // Miru defaults
    const child = spawn(path.join(ROOT, "node_modules", ".bin", "bun"), ["scripts/miru_depth_ablation.ts", INDEX, queriesPath, vecPath, rawPath],
      { cwd: ROOT, env, stdio: ["ignore", "inherit", "inherit"] });
    child.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`bun exited with ${code}`))));
  });
}

// 2. Scoring -----------------------------------------------------------------------------------------------------
interface Trace { file: string; line: number; denseRank: number | null; bm25Rank: number | null; fused: number; boosted: number }
interface Row { id: string; top: { file: string; line: number }[]; pool: number; trace?: Trace[] }
const raw = JSON.parse(readFileSync(rawPath, "utf-8")) as {
  runtime: string; chunks: number; miru: Record<string, number>; locationQueries: string[];
  fidelity: Record<string, { identical: number; total: number; mismatches: string[] }>;
  configs: { id: string; group: string; denseDepth: number; bm25Depth: number; topK: number; alpha?: number | null; rerank?: boolean; env?: Record<string, string> }[];
  results: Record<string, Row[]>;
};
const n = spec.queries.length;
const files = (r: Row) => r.top.map((t) => t.file);
const hitVec = (id: string) => raw.results[id].map((r) => hitAt3(files(r), byId.get(r.id)!.definedIn));
const rrVec = (id: string) => raw.results[id].map((r) => rrAt3(files(r), byId.get(r.id)!.definedIn));
const summary = (id: string) => {
  const h = hitVec(id), k = h.filter(Boolean).length, [lo, hi] = wilson(k, n);
  const rr = rrVec(id);
  return { id, hits: k, hitAt3: k / n, ci95: [lo, hi] as [number, number], mrrAt3: rr.reduce((a, b) => a + b, 0) / n, meanPool: raw.results[id].reduce((a, r) => a + r.pool, 0) / n };
};
const compare = (a: string, b: string) => {
  const ha = hitVec(a), hb = hitVec(b);
  const pr = paired(ha, hb);
  const [lo, hi] = bootstrapDeltaCI(ha.map(Number), hb.map(Number));
  return { delta: (ha.filter(Boolean).length - hb.filter(Boolean).length) / n, ci95: [lo, hi] as [number, number], wins: pr.wins, losses: pr.losses, p: pr.p };
};

// formatting helpers
const pct = (x: number) => `${(100 * x).toFixed(1)}%`;
const pts = (x: number) => `${x >= 0 ? "+" : ""}${(100 * x).toFixed(1)}`;
const ci = (c: [number, number]) => `[${(100 * c[0]).toFixed(1)}, ${(100 * c[1]).toFixed(1)}]`;
const pv = (p: number) => (p < 0.001 ? "<0.001" : p.toFixed(3));
const table = (title: string, head: string[], body: string[][]) => {
  const w = head.map((h, i) => Math.max(h.length, ...body.map((r) => r[i].length)));
  const line = (l: string, m: string, r: string) => l + w.map((x) => "─".repeat(x + 2)).join(m) + r;
  const row = (r: string[]) => "│ " + r.map((s, i) => (i === 0 ? s.padEnd(w[i]) : s.padStart(w[i]))).join(" │ ") + " │";
  console.log(`\n${title}\n${[line("┌", "┬", "┐"), row(head), line("├", "┼", "┤"), ...body.map(row), line("└", "┴", "┘")].join("\n")}`);
};
const out: Record<string, unknown> = { repo: spec.repo, commit: spec.commit, chunks: raw.chunks, queries: n, embedder: emb.name, runtime: raw.runtime, miru: raw.miru, fidelity: raw.fidelity, locationQueries: raw.locationQueries };

console.log(`\nMiru candidate-depth ablation · ${spec.repo}@${spec.commit.slice(0, 7)} · ${raw.chunks.toLocaleString("en-US")} chunks · ${n} queries · ${emb.name} · ${raw.runtime}`);
console.log(`Fidelity vs MiruIndex.search: ${Object.entries(raw.fidelity).map(([k, f]) => `${k} ${f.identical}/${f.total}`).join(" · ")}`);

// E1 sweep
const sweep = raw.configs.filter((c) => c.group === "sweep");
const ref250 = "sym_d250";
out.sweep = sweep.map((c) => ({ depth: c.denseDepth, ...summary(c.id), vsDepth250: c.id === ref250 ? null : compare(c.id, ref250) }));
table("E1 · Symmetric candidate depth (both lists), final topK = 3", ["Depth / list", "Pool", "Hit@3", "95% CI", "MRR@3", "Δ vs 250", "95% CI Δ", "W/L", "p"],
  sweep.map((c) => { const s = summary(c.id), d = c.id === ref250 ? null : compare(c.id, ref250);
    return [String(c.denseDepth), s.meanPool.toFixed(0), pct(s.hitAt3), ci(s.ci95), s.mrrAt3.toFixed(3), d ? pts(d.delta) : "—", d ? ci(d.ci95) : "—", d ? `${d.wins}/${d.losses}` : "—", d ? pv(d.p) : "—"]; }));

// E2 grid
const G = [...new Set(raw.configs.filter((c) => c.group === "grid").map((c) => c.denseDepth))];
out.grid = raw.configs.filter((c) => c.group === "grid").map((c) => ({ denseDepth: c.denseDepth, bm25Depth: c.bm25Depth, ...summary(c.id) }));
table("E2 · Hit@3 by dense depth (rows) × BM25 depth (columns), final topK = 3", ["Dense \\ BM25", ...G.map(String)],
  G.map((D) => [String(D), ...G.map((B) => { const s = summary(`grid_D${D}_B${B}`); return `${pct(s.hitAt3)} (${s.hits})`; })]));
const gridEffect = (fixed: "dense" | "bm25") => G.map((v) => {
  const shallow = fixed === "dense" ? `grid_D${v}_B${G[0]}` : `grid_D${G[0]}_B${v}`;
  const deep = fixed === "dense" ? `grid_D${v}_B${G[G.length - 1]}` : `grid_D${G[G.length - 1]}_B${v}`;
  return { fixedAt: v, ...compare(shallow, deep) };
});
out.gridEffects = { deepenBm25AtFixedDense: gridEffect("dense"), deepenDenseAtFixedBm25: gridEffect("bm25") };
table(`E2b · Effect of deepening ONE list from ${G[0]} → ${G[G.length - 1]} (shallow minus deep; positive = deepening hurts)`, ["Other list fixed at", "Deepen BM25 only", "W/L", "p", "Deepen dense only", "W/L", "p"],
  G.map((v, i) => { const b = (out.gridEffects as any).deepenBm25AtFixedDense[i], d = (out.gridEffects as any).deepenDenseAtFixedBm25[i];
    return [String(v), pts(b.delta), `${b.wins}/${b.losses}`, pv(b.p), pts(d.delta), `${d.wins}/${d.losses}`, pv(d.p)]; }));

// E3 factorial
const F = { d15k3: "fact_d15_k3", d250k3: "fact_d250_k3", d15k50: "fact_d15_k50", d250k50: "fact_d250_k50" };
out.factorial = {
  cells: Object.fromEntries(Object.entries(F).map(([k, id]) => [k, summary(id)])),
  depthEffectAtTopK3: compare(F.d15k3, F.d250k3), depthEffectAtTopK50: compare(F.d15k50, F.d250k50),
  topKEffectAtDepth15: compare(F.d15k3, F.d15k50), topKEffectAtDepth250: compare(F.d250k3, F.d250k50),
  optimizedVsLegacy: compare(F.d15k3, F.d250k50),
};
table("E3 · 2×2 factorial: candidate depth × final rerank topK (Hit@3 of first 3 results)", ["Depth / list", "final topK = 3", "final topK = 50"],
  [15, 250].map((d) => [String(d), pct(summary(`fact_d${d}_k3`).hitAt3), pct(summary(`fact_d${d}_k50`).hitAt3)]));
const fx = out.factorial as Record<string, any>;
table("E3b · Simple effects", ["Contrast", "Δ Hit@3", "95% CI", "W/L", "p"], [
  ["Depth 15 vs 250 (topK 3)", fx.depthEffectAtTopK3], ["Depth 15 vs 250 (topK 50)", fx.depthEffectAtTopK50],
  ["topK 3 vs 50 (depth 15)", fx.topKEffectAtDepth15], ["topK 3 vs 50 (depth 250)", fx.topKEffectAtDepth250],
  ["Optimized vs Legacy", fx.optimizedVsLegacy],
].map(([l, c]: any) => [l, pts(c.delta), ci(c.ci95), `${c.wins}/${c.losses}`, pv(c.p)]));

// E4 alpha
const alphas = [...new Set(raw.configs.filter((c) => c.group === "alpha").map((c) => (c.alpha === null || c.alpha === undefined ? "default" : String(c.alpha))))];
out.alpha = alphas.map((a) => ({ alpha: a, d15: summary(`alpha_${a}_d15`), d250: summary(`alpha_${a}_d250`), depthEffect: compare(`alpha_${a}_d15`, `alpha_${a}_d250`) }));
table("E4 · Fusion weight α (0 = BM25 only, default = 0.25 for these queries, 1 = dense only)", ["α", "Depth 15", "Depth 250", "Δ (15 − 250)", "W/L", "p"],
  (out.alpha as any[]).map((r) => [r.alpha, pct(r.d15.hitAt3), pct(r.d250.hitAt3), pts(r.depthEffect.delta), `${r.depthEffect.wins}/${r.depthEffect.losses}`, pv(r.depthEffect.p)]));

// E5 stage ablations
const stages = [["full", "Full Miru pipeline"], ["nostem", "No path-stem boosts (EXACT_STEM_BOOST=0)"], ["nopenalty", "No path penalties"], ["fusiononly", "Fusion only (no boosts, no rerank)"]];
out.stages = stages.map(([s, label]) => ({ stage: s, label, d15: summary(`stage_${s}_d15`), d250: summary(`stage_${s}_d250`), depthEffect: compare(`stage_${s}_d15`, `stage_${s}_d250`) }));
table("E5 · Post-fusion stage ablations", ["Pipeline", "Depth 15", "Depth 250", "Δ (15 − 250)", "95% CI", "W/L", "p"],
  (out.stages as any[]).map((r) => [r.label, pct(r.d15.hitAt3), pct(r.d250.hitAt3), pts(r.depthEffect.delta), ci(r.depthEffect.ci95), `${r.depthEffect.wins}/${r.depthEffect.losses}`, pv(r.depthEffect.p)]));

// E6 dose-response
const betas = [...new Set(raw.configs.filter((c) => c.group === "beta").map((c) => c.env!.MIRU_RANK_EXACT_STEM_BOOST))];
out.beta = betas.map((b) => ({ beta: Number(b), d15: summary(`beta_${b}_d15`), d250: summary(`beta_${b}_d250`), depthEffect: compare(`beta_${b}_d15`, `beta_${b}_d250`) }));
table("E6 · Dose-response: path-stem boost strength β (Miru default 0.75)", ["β", "Depth 15", "Depth 250", "Δ (15 − 250)", "95% CI", "W/L", "p"],
  (out.beta as any[]).map((r) => [String(r.beta), pct(r.d15.hitAt3), pct(r.d250.hitAt3), pts(r.depthEffect.delta), ci(r.depthEffect.ci95), `${r.depthEffect.wins}/${r.depthEffect.losses}`, pv(r.depthEffect.p)]));

// 3. Displacement attribution on queries Legacy misses but Optimized hits ----------------------------------------
const legacyRows = new Map(raw.results[F.d250k50].map((r) => [r.id, r]));
const optRows = new Map(raw.results[F.d15k3].map((r) => [r.id, r]));
const flipped = spec.queries.filter((q) => !hitAt3(files(legacyRows.get(q.id)!), q.definedIn) && hitAt3(files(optRows.get(q.id)!), q.definedIn));
const SHALLOW = 15;
const origin = (t: Trace) => {
  const d = t.denseRank !== null, b = t.bm25Rank !== null;
  const dS = d && t.denseRank! <= SHALLOW, bS = b && t.bm25Rank! <= SHALLOW;
  const inShallow = dS || bS;
  const lists = d && b ? "both lists" : d ? "dense only" : "BM25 only";
  const admittedBy = inShallow ? "already in depth-15 pool" : d && b ? "deep tail of both lists" : d ? "deep tail of dense list" : "deep tail of BM25 list";
  return { lists, admittedBy, inShallow };
};
const base = (f: string) => path.basename(f).replace(/^fake_/, "");
const perQuery = flipped.map((q) => {
  const L = legacyRows.get(q.id)!, O = optRows.get(q.id)!;
  const key = (x: { file: string; line: number }) => `${x.file}:${x.line}`;
  const traceL = new Map(L.trace!.map((t) => [key(t), t]));
  const traceO = new Map(O.trace!.map((t) => [key(t), t]));
  const words = new Set(q.query.toLowerCase().split(/\s+/));
  const intruders = L.top.slice(0, 3).map((t) => {
    const tr = traceL.get(key(t))!, inO = traceO.get(key(t));
    // How did the deep pool help this result? Not in the shallow pool at all → admission; in it, but with a higher
    // fused (pre-boost) score in the deep pool → it collected RRF credit from a second list.
    const pathway = !inO ? "admission (absent from depth-15 pool)" : tr.fused > inO.fused + 1e-12 ? "accumulation (extra RRF credit from 2nd list)" : "present at equal fused score";
    const stem = path.basename(t.file).replace(/\.[^.]+$/, "").toLowerCase();
    return { file: t.file, ...origin(tr), pathway, denseRank: tr.denseRank, bm25Rank: tr.bm25Rank, fused: tr.fused, fusedInOptimizedPool: inO?.fused ?? null, boost: tr.boosted - tr.fused,
      exactStemMatch: words.has(stem), sameBasenameAsTarget: q.definedIn.some((d) => base(d) === base(t.file)) };
  });
  const tgtL = L.trace!.filter((t) => q.definedIn.includes(t.file)).sort((a, b) => b.boosted - a.boosted)[0] ?? null;
  const tgtO = O.trace!.filter((t) => q.definedIn.includes(t.file)).sort((a, b) => b.boosted - a.boosted)[0] ?? null;
  return { id: q.id, query: q.query, intruders, targetInLegacy: tgtL, targetInOptimized: tgtO };
});
const allIntr = perQuery.flatMap((p) => p.intruders);
const countBy = <T,>(xs: T[], f: (x: T) => string) => xs.reduce<Record<string, number>>((m, x) => ((m[f(x)] = (m[f(x)] ?? 0) + 1), m), {});
const deepIntr = allIntr.filter((i) => !i.inShallow);
const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.floor(s.length / 2)] : Number.NaN; };
const targetBoosts = perQuery.flatMap((p) => (p.targetInLegacy ? [p.targetInLegacy.boosted - p.targetInLegacy.fused] : []));
const stemHits = (d: number) => compare(`stage_full_d${d}`, `stage_nostem_d${d}`);
out.attribution = {
  flippedQueries: flipped.length, intruders: allIntr.length,
  byPathway: countBy(allIntr, (i) => i.pathway), byAdmission: countBy(allIntr, (i) => i.admittedBy), byLists: countBy(allIntr, (i) => i.lists),
  deepIntruders: deepIntr.length, deepIntrudersWithStemBoost: deepIntr.filter((i) => i.boost > 0).length,
  intrudersWithStemBoost: allIntr.filter((i) => i.boost > 0).length, intrudersExactStemMatch: allIntr.filter((i) => i.exactStemMatch).length,
  exactStems: countBy(allIntr.filter((i) => i.exactStemMatch), (i) => path.basename(i.file)),
  intrudersSameBasenameAsTarget: allIntr.filter((i) => i.sameBasenameAsTarget).length,
  medianIntruderBoost: median(allIntr.map((i) => i.boost)), medianIntruderFused: median(allIntr.map((i) => i.fused)), medianTargetBoost: median(targetBoosts),
  targetFusedHigherInOptimized: perQuery.filter((p) => p.targetInLegacy && p.targetInOptimized && p.targetInOptimized.fused > p.targetInLegacy.fused).length,
  stemBoostEffect: { depth15: stemHits(15), depth250: stemHits(250) },
  perQuery,
};
const at = out.attribution as Record<string, any>;
table(`Attribution · ${flipped.length} queries Legacy misses and Optimized hits → ${allIntr.length} Legacy top-3 results that displaced the target`, ["How the deep pool helped the displacing result", "Count"],
  Object.entries(at.byPathway).sort((a: any, b: any) => b[1] - a[1]).map(([k, v]) => [k, String(v)]));
console.log(`  • ${at.intrudersWithStemBoost}/${allIntr.length} received a path-stem boost; ${at.intrudersExactStemMatch}/${allIntr.length} have a file stem that equals a query word ${JSON.stringify(at.exactStems)}.`);
console.log(`  • Median boost: displacing result ${at.medianIntruderBoost.toFixed(4)} (its fused score ${at.medianIntruderFused.toFixed(4)}) vs target ${at.medianTargetBoost.toFixed(4)}.`);
console.log(`  • List membership: ${JSON.stringify(at.byLists)}; pool entry: ${JSON.stringify(at.byAdmission)}.`);
console.log(`  • Stem boost on vs off (W/L in Hit@3): depth 15 → ${at.stemBoostEffect.depth15.wins}/${at.stemBoostEffect.depth15.losses} (${pts(at.stemBoostEffect.depth15.delta)} pts), depth 250 → ${at.stemBoostEffect.depth250.wins}/${at.stemBoostEffect.depth250.losses} (${pts(at.stemBoostEffect.depth250.delta)} pts).`);

const outFile = path.join(ROOT, "benchmark", `results.ablation.${emb.name}.json`);
writeFileSync(outFile, JSON.stringify(out, null, 2));
console.log(`\nResults → ${path.relative(ROOT, outFile)}`);
