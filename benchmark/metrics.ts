/** Pure metric helpers shared by benchmark/topk_eval.ts and tests/run.ts. */

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
