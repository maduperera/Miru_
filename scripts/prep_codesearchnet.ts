/**
 * Rebuilds Miru's own CodeSearchNet benchmark corpus (miru-code scripts/benchmark-codesearchnet-compression.ts)
 * in the shape our Kalpana pipeline expects: one corpus per language so results split cleanly by language.
 *
 *   npx tsx scripts/prep_codesearchnet.ts [--langs go,java,javascript,php,python,ruby] [--n 500]
 *
 * Same sampling as Miru: dataset code-search-net/code_search_net, split=test, first N rows per language that
 * have non-empty code + docstring, one function per file, query = func_documentation_string.
 * Writes:
 *   eval-repos/csn-<lang>/<NNNN>.<ext>          (function source, exactly as Miru writes it)
 *   benchmark/queries.csn-<lang>.json            ({id, query, identifier, definedIn}) for run_lang.sh
 * `identifier` = last segment of func_name (e.g. "Foo.bar" -> "bar"), so Def@3 = the chunk tree-sitter
 * says defines that function.
 */
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DATASET = "code-search-net/code_search_net";
const EXT: Record<string, string> = { go: "go", java: "java", javascript: "js", php: "php", python: "py", ruby: "rb" };
const args = process.argv.slice(2);
const flag = (n: string, d: string) => (args.includes(n) ? args[args.indexOf(n) + 1] : d);
const LANGS = flag("--langs", Object.keys(EXT).join(",")).split(",");
const N = Number(flag("--n", "500"));
const PAGE = 100;

interface Row { func_name: string; func_code_string: string; func_documentation_string: string; repository_name: string }

async function page(lang: string, offset: number): Promise<Row[]> {
  const url = new URL("https://datasets-server.huggingface.co/rows");
  Object.entries({ dataset: DATASET, config: lang, split: "test", offset: String(offset), length: String(PAGE) })
    .forEach(([k, v]) => url.searchParams.set(k, v));
  for (let attempt = 0; ; attempt++) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const body = (await r.json()) as { rows?: { row: Row }[] };
      return (body.rows ?? []).map((e) => e.row);
    } catch (e) {
      if (attempt >= 4) throw new Error(`CodeSearchNet ${lang}@${offset}: ${e}`);
      await new Promise((s) => setTimeout(s, 2000 * (attempt + 1)));
    }
  }
}

for (const lang of LANGS) {
  const rows: Row[] = [];
  // Miru filters inside each page and slices to N; mirror that (pages until N usable rows).
  for (let off = 0; rows.length < N; off += PAGE) {
    const got = await page(lang, off);
    if (!got.length) break;
    rows.push(...got.filter((r) => r.func_code_string?.trim() && r.func_documentation_string?.trim()));
  }
  if (rows.length < N) throw new Error(`${lang}: only ${rows.length}/${N} usable rows`);
  const repo = `csn-${lang}`;
  const dir = path.join(ROOT, "eval-repos", repo);
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  const queries = rows.slice(0, N).map((r, i) => {
    const file = `${String(i).padStart(4, "0")}.${EXT[lang]}`;
    writeFileSync(path.join(dir, file), `${r.func_code_string.trimEnd()}\n`);
    const identifier = r.func_name.split(/::|\.|->|#/).filter(Boolean).pop() ?? r.func_name;
    return { id: `CSN_${lang.toUpperCase()}${i}`, query: r.func_documentation_string.trim(), identifier,
      definedIn: [file], source: r.repository_name };
  });
  writeFileSync(path.join(ROOT, "benchmark", `queries.${repo}.json`), JSON.stringify({
    repo, dataset: DATASET, split: "test", metric: "defining file / defining chunk in top 3",
    description: `Miru's CodeSearchNet protocol: first ${N} usable test rows, docstring -> function, one function per file.`,
    queries,
  }, null, 1));
  console.log(`${lang}: ${queries.length} functions -> eval-repos/${repo}, benchmark/queries.${repo}.json`);
}
