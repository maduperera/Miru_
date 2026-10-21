/**
 * Turns Miru's curated cross-repo A/B suite (miru-code scripts/search-ab-queries.ts, used by
 * search-ab-multi.ts) into our {id, query, identifier, definedIn} format and clones the repos.
 *
 *   npx tsx scripts/prep_miru_ab.ts [--no-clone]
 *
 * Miru checks these out unpinned under ~/.cache/miru-bench; we shallow-clone HEAD into eval-repos/ab-<name>
 * and RECORD the commit so our run is reproducible. These queries are labelled at FILE level only
 * (`relevant` files, no symbol), so only File@3 / MRR(file) are meaningful; identifier is "".
 */
import { execFileSync } from "node:child_process";
import { existsSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { REPO_BENCHES } from "../eval-repos/miru-code/scripts/search-ab-queries.ts";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const URLS: Record<string, string> = {
  "miru-code": "https://github.com/takara-ai/miru-code",
  flask: "https://github.com/pallets/flask",
  fastapi: "https://github.com/fastapi/fastapi",
  click: "https://github.com/pallets/click",
  axum: "https://github.com/tokio-rs/axum",
  serde: "https://github.com/serde-rs/serde",
  gin: "https://github.com/gin-gonic/gin",
  cobra: "https://github.com/spf13/cobra",
  express: "https://github.com/expressjs/express",
  phoenix: "https://github.com/phoenixframework/phoenix",
  rails: "https://github.com/rails/rails",
};
const clone = !process.argv.includes("--no-clone");
const git = (...a: string[]) => execFileSync("git", a, { encoding: "utf8" }).trim();

for (const b of REPO_BENCHES) {
  const repo = `ab-${b.name}`;
  const dir = path.join(ROOT, "eval-repos", repo);
  if (clone && !existsSync(dir)) {
    if (!URLS[b.name]) { console.warn(`skip ${b.name}: no URL`); continue; }
    git("clone", "--depth", "1", "--quiet", URLS[b.name], dir);
  }
  if (!existsSync(dir)) { console.warn(`skip ${b.name}: not cloned`); continue; }
  const commit = git("-C", dir, "rev-parse", "HEAD");
  const queries = b.queries.map((q, i) => ({
    id: `AB_${b.name.toUpperCase().replace(/\W/g, "")}${i}`, query: q.query, identifier: "",
    definedIn: q.relevant, category: q.category,
  }));
  writeFileSync(path.join(ROOT, "benchmark", `queries.${repo}.json`), JSON.stringify({
    repo, language: b.language, commit, source: "miru-code scripts/search-ab-queries.ts",
    metric: "relevant file in top 3 (file-level labels only)", queries,
  }, null, 1));
  console.log(`${b.name} (${b.language}) @ ${commit.slice(0, 10)}: ${queries.length} queries`);
}
