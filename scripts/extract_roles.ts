/**
 * Language-agnostic Kalpana role extractor built on the SAME tree-sitter grammars Miru vendors
 * (@takara-ai/miru-code/grammars, web-tree-sitter runtime). Replaces the Go-only regex extractor.
 *
 *   npx tsx scripts/extract_roles.ts pool <candidates.json> <out.json>
 *       Adds `roles` ([role, rawIdentifier][]) and `defs` (identifiers defined in the chunk) to every
 *       candidate of a Miru candidates dump. Python normalises roles exactly like go_pairs().
 *   npx tsx scripts/extract_roles.ts repo <repoDir> <out.json> [--lang go,python,...]
 *       Walks a repository and writes { identifier: [files that define it] } for query generation.
 *
 * Role mapping (one table for every language):
 *   ROLE_CLASS  class/struct/interface/enum/trait/type/module declarations
 *   ROLE_FUNC   function/method/constructor declarations (bodies defined in the chunk)
 *   ROLE_ARG    parameter names of those functions
 *   ROLE_VAR    assignment / declaration targets
 *   ROLE_CALL   called function/method name (last segment), excluding names defined in the chunk
 */
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Language, Parser, type Node } from "web-tree-sitter";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const GRAMMARS = path.join(ROOT, "node_modules", "@takara-ai", "miru-code", "grammars");
const manifest = JSON.parse(readFileSync(path.join(GRAMMARS, "manifest.json"), "utf8")) as { languages: Record<string, string> };

export const EXT_LANG: Record<string, string> = {
  ".go": "go", ".py": "python", ".java": "java", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
  ".ts": "typescript", ".tsx": "typescript", ".rs": "rust", ".rb": "ruby", ".php": "php", ".cs": "csharp",
  ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".scala": "scala", ".dart": "dart",
  ".ex": "elixir", ".exs": "elixir", ".hs": "haskell", ".ml": "ocaml", ".sol": "solidity",
};

const CLASS_TYPES = new Set([
  "class_declaration", "class_definition", "class", "module", "interface_declaration", "enum_declaration",
  "struct_item", "enum_item", "trait_item", "type_item", "union_item", "type_spec", "type_alias",
  "type_alias_declaration", "record_declaration", "struct_declaration", "struct_specifier", "class_specifier",
  "trait_declaration", "object_definition", "trait_definition", "abstract_class_declaration", "enum_specifier",
  "protocol_declaration", "contract_declaration",
]);
const FUNC_TYPES = new Set([
  "function_declaration", "function_definition", "method_declaration", "method_definition", "function_item",
  "method", "singleton_method", "constructor_declaration", "generator_function_declaration",
  "local_function_statement", "function_signature_item", "method_signature",
]);
const CALL_TYPES = new Set(["call_expression", "call", "method_invocation", "invocation_expression",
  "function_call_expression", "member_call_expression", "scoped_call_expression"]);
const PARAM_CONTAINERS = ["parameters", "parameter_list", "formal_parameters", "parameter"];
const KEYWORDS = new Set(("if for switch select return func go defer range make new len cap append panic recover copy " +
  "delete close map chan struct interface type var const package import else case default break continue goto " +
  "fallthrough nil true false string int bool error byte rune self this super print require").split(" "));

const IDENT = /^(identifier|type_identifier|field_identifier|property_identifier|constant|name|simple_identifier|shorthand_property_identifier_pattern)$/;

function lastIdent(n: Node | null): string | null {
  if (!n) return null;
  if (IDENT.test(n.type)) return n.text;
  for (const f of ["field", "property", "name", "attribute", "method"]) {
    const c = n.childForFieldName(f);
    if (c) return lastIdent(c);
  }
  const kids = n.namedChildren.filter(Boolean) as Node[];
  for (let i = kids.length - 1; i >= 0; i--) {
    const r = lastIdent(kids[i]);
    if (r) return r;
  }
  return null;
}

function firstIdent(n: Node | null): string | null {
  if (!n) return null;
  if (IDENT.test(n.type)) return n.text;
  for (const c of n.namedChildren) {
    const r = c && firstIdent(c);
    if (r) return r;
  }
  return null;
}

function defName(n: Node): string | null {
  const nm = n.childForFieldName("name");
  if (nm) return lastIdent(nm);
  if (n.type === "struct_specifier" || n.type === "class_specifier" || n.type === "enum_specifier") return null;
  const decl = n.childForFieldName("declarator"); // C/C++ function_definition
  return decl ? firstIdent(decl) : null;
}

function params(fn: Node): string[] {
  let box: Node | null = null;
  for (const f of PARAM_CONTAINERS) if ((box = fn.childForFieldName(f))) break;
  if (!box) {
    const decl = fn.childForFieldName("declarator");
    box = decl?.childForFieldName("parameters") ?? null;
  }
  if (!box) return [];
  const out: string[] = [];
  for (const p of box.namedChildren) {
    if (!p) continue;
    if (IDENT.test(p.type)) { out.push(p.text); continue; }        // python bare param
    const names = p.childrenForFieldName("name");
    if (names.length) { for (const x of names) if (x) out.push(lastIdent(x) ?? x.text); continue; }
    const pat = p.childForFieldName("pattern") ?? p.childForFieldName("declarator");
    const id = pat ? firstIdent(pat) : firstIdent(p);
    if (id) out.push(id);
  }
  return out;
}

function varTargets(n: Node): string[] {
  switch (n.type) {
    case "short_var_declaration": case "assignment_statement": {
      const left = n.childForFieldName("left");
      return left ? (left.namedChildren.filter((c) => c && c.type === "identifier") as Node[]).map((c) => c.text) : [];
    }
    case "var_spec": case "const_spec":
      return (n.childrenForFieldName("name").filter(Boolean) as Node[]).map((c) => c.text);
    case "assignment": case "augmented_assignment": {             // python, ruby
      const left = n.childForFieldName("left");
      return left && left.type === "identifier" ? [left.text] : [];
    }
    case "variable_declarator": {                                  // js/ts/java/c#
      const nm = n.childForFieldName("name");
      return nm && IDENT.test(nm.type) ? [nm.text] : [];
    }
    case "let_declaration": {                                      // rust
      const p = n.childForFieldName("pattern");
      return p && p.type === "identifier" ? [p.text] : [];
    }
    default:
      return [];
  }
}

export interface Extracted { roles: [string, string][]; defs: string[] }

export function extract(root: Node): Extracted {
  const defsFunc: string[] = [], defsClass: string[] = [], args: string[] = [], vars: string[] = [], calls: string[] = [];
  const stack: Node[] = [root];
  while (stack.length) {   // pre-order, document order (stack reversed below)
    const n = stack.pop()!;
    if (FUNC_TYPES.has(n.type)) {
      const nm = defName(n);
      if (nm) { defsFunc.push(nm); args.push(...params(n)); }
    } else if (CLASS_TYPES.has(n.type)) {
      const nm = defName(n);
      if (nm) defsClass.push(nm);
    } else if (CALL_TYPES.has(n.type)) {
      const f = n.childForFieldName("function") ?? n.childForFieldName("name") ?? n.childForFieldName("method");
      const nm = n.type === "call" && n.childForFieldName("method") ? lastIdent(n.childForFieldName("method")) : lastIdent(f);
      if (nm) calls.push(nm);
    } else {
      vars.push(...varTargets(n));
    }
    const kids = n.namedChildren;
    for (let i = kids.length - 1; i >= 0; i--) if (kids[i]) stack.push(kids[i]!);
  }
  const defined = new Set([...defsFunc, ...defsClass]);
  // Same ordering contract as the Go regex extractor: FUNC(+ARG), CLASS, VAR, CALL.
  const roles: [string, string][] = [];
  for (const f of defsFunc) roles.push(["ROLE_FUNC", f]);
  for (const a of args) roles.push(["ROLE_ARG", a]);
  for (const c of defsClass) roles.push(["ROLE_CLASS", c]);
  for (const v of vars) if (v !== "_" && !KEYWORDS.has(v)) roles.push(["ROLE_VAR", v]);
  for (const c of calls) if (!KEYWORDS.has(c) && !defined.has(c)) roles.push(["ROLE_CALL", c]);
  return { roles, defs: [...defined] };
}

// ---------------------------------------------------------------------------------------------
let parser: Parser | null = null;
const langCache = new Map<string, Language>();

async function parserFor(language: string, filePath = ""): Promise<Parser | null> {
  const file = language === "typescript" && filePath.endsWith(".tsx") ? "tree-sitter-tsx.wasm" : manifest.languages[language];
  if (!file || !existsSync(path.join(GRAMMARS, file))) return null;
  if (!parser) {
    await Parser.init({ locateFile: () => path.join(ROOT, "node_modules", "web-tree-sitter", "web-tree-sitter.wasm") });
    parser = new Parser();
  }
  let lang = langCache.get(file);
  if (!lang) { lang = await Language.load(path.join(GRAMMARS, file)); langCache.set(file, lang); }
  parser.setLanguage(lang);
  return parser;
}

export async function extractSource(src: string, language: string | null, filePath = ""): Promise<Extracted | null> {
  const lang = language ?? EXT_LANG[path.extname(filePath)] ?? null;
  if (!lang) return null;
  const p = await parserFor(lang, filePath);
  if (!p) return null;
  const tree = p.parse(src);
  if (!tree) return null;
  try { return extract(tree.rootNode); } finally { tree.delete(); }
}

async function modePool(inPath: string, outPath: string) {
  const data = JSON.parse(readFileSync(inPath, "utf8"));
  let n = 0, ok = 0;
  const byLang: Record<string, number> = {};
  for (const q of data.queries) {
    for (const c of q.candidates) {
      n++;
      const ex = await extractSource(c.content, c.language, c.file_path);
      c.roles = ex?.roles ?? [];
      c.defs = ex?.defs ?? [];
      if (c.roles.length) ok++;
      const l = c.language ?? "?"; byLang[l] = (byLang[l] ?? 0) + 1;
    }
  }
  data.extractor = "tree-sitter (miru grammars)";
  writeFileSync(outPath, JSON.stringify(data));
  console.log(`roles for ${n} candidates (${((100 * ok) / n).toFixed(1)}% with >=1 role) by language ${JSON.stringify(byLang)} -> ${outPath}`);
}

function walk(dir: string, out: string[]) {
  for (const e of readdirSync(dir)) {
    if (e === ".git" || e === "node_modules" || e === "vendor") continue;
    const p = path.join(dir, e);
    const s = statSync(p);
    if (s.isDirectory()) walk(p, out);
    else if (EXT_LANG[path.extname(e)]) out.push(p);
  }
}

async function modeRepo(repoDir: string, outPath: string, langs: Set<string> | null) {
  const files: string[] = [];
  walk(repoDir, files);
  const defs: Record<string, string[]> = {};
  let parsed = 0;
  for (const f of files) {
    const lang = EXT_LANG[path.extname(f)];
    if (langs && !langs.has(lang)) continue;
    const src = readFileSync(f, "utf8");
    if (src.length > 1_000_000) continue;
    const ex = await extractSource(src, lang, f);
    if (!ex) continue;
    parsed++;
    const rel = path.relative(repoDir, f).split(path.sep).join("/");
    for (const d of new Set(ex.defs)) (defs[d] ??= []).push(rel);
  }
  writeFileSync(outPath, JSON.stringify({ repoDir: path.relative(ROOT, repoDir), files: parsed, defs }));
  console.log(`parsed ${parsed} files, ${Object.keys(defs).length} defined identifiers -> ${outPath}`);
}

const [mode, a, b, ...rest] = process.argv.slice(2);
if (mode === "pool" && a && b) await modePool(a, b);
else if (mode === "repo" && a && b) {
  const i = rest.indexOf("--lang");
  await modeRepo(path.resolve(a), b, i >= 0 ? new Set(rest[i + 1].split(",")) : null);
} else if (mode) {
  console.error("usage: extract_roles.ts pool <candidates.json> <out.json> | repo <repoDir> <out.json> [--lang go,python]");
  process.exit(1);
}
