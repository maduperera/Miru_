/** Pure metric helpers shared by benchmark/topk_eval.ts, benchmark/depth_ablation.ts and tests/run.ts. */

/** True when any of the first 3 result files is one of the target files. */
export function hitAt3(resultFiles: readonly string[], targets: readonly string[]): boolean {
  return resultFiles.slice(0, 3).some((f) => targets.includes(f));
}

/** Nearest-rank percentile (p in 0–100). */
export function percentile(xs: readonly number[], p: number): number {
  if (!xs.length) return Number.NaN;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.ceil((p / 100) * s.length) - 1))];
}

/** Exact two-sided sign test on paired outcomes: `wins` vs `losses` (ties discarded). */
export function signTestP(wins: number, losses: number): number {
  const n = wins + losses;
  if (n === 0) return 1;
  let tail = 0;
  let c = 1; // C(n, 0)
  for (let k = 0; k <= Math.min(wins, losses); k++) {
    tail += c;
    c = (c * (n - k)) / (k + 1);
  }
  return Math.min(1, (2 * tail) / 2 ** n);
}

/** Paired wins/losses of arm A over arm B on per-query hit vectors. */
export function paired(a: readonly boolean[], b: readonly boolean[]): { wins: number; losses: number; p: number } {
  let wins = 0, losses = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] && !b[i]) wins++;
    else if (!a[i] && b[i]) losses++;
  }
  return { wins, losses, p: signTestP(wins, losses) };
}

/** Reciprocal rank of the first target file within the first 3 results (0 if none). */
export function rrAt3(resultFiles: readonly string[], targets: readonly string[]): number {
  const i = resultFiles.slice(0, 3).findIndex((f) => targets.includes(f));
  return i < 0 ? 0 : 1 / (i + 1);
}

/** Wilson score 95% interval for a binomial proportion k/n. */
export function wilson(k: number, n: number, z = 1.959964): [number, number] {
  if (n === 0) return [Number.NaN, Number.NaN];
  const p = k / n;
  const d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [c - h, c + h];
}

/** Deterministic PRNG (mulberry32) so bootstrap intervals are reproducible. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Paired percentile-bootstrap 95% CI for mean(a) - mean(b) over queries (a, b are per-query scores). */
export function bootstrapDeltaCI(a: readonly number[], b: readonly number[], iters = 10000, seed = 42): [number, number] {
  const n = a.length;
  const rand = mulberry32(seed);
  const deltas: number[] = [];
  for (let it = 0; it < iters; it++) {
    let s = 0;
    for (let j = 0; j < n; j++) {
      const i = Math.floor(rand() * n);
      s += a[i] - b[i];
    }
    deltas.push(s / n);
  }
  deltas.sort((x, y) => x - y);
  return [deltas[Math.floor(0.025 * iters)], deltas[Math.ceil(0.975 * iters) - 1]];
}
