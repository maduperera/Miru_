// Offline smoke test: does extract_roles.ts produce sane roles for every Miru benchmark language?
//   node --import tsx tests/roles_smoke.ts
import { extractSource } from "../scripts/extract_roles.ts";

const S: Record<string, [string, string, string]> = {
  go: ["a.go", "HandleRequest", "package x\ntype Server struct{}\nfunc (s *Server) HandleRequest(ctx Context, req *Req) error {\n  resp := s.route(req)\n  return writeBody(resp)\n}"],
  java: ["A.java", "findUser", "public class UserService {\n  public User findUser(String userId, int depth) {\n    User u = repo.lookup(userId);\n    return validate(u);\n  }\n}"],
  javascript: ["a.js", "createApplication", "class Router {}\nfunction createApplication(opts, cb) {\n  const app = mixin(opts);\n  return app.init(cb);\n}"],
  typescript: ["a.ts", "buildIndex", "interface Opts { a: number }\nexport function buildIndex(repoPath: string, opts: Opts): void {\n  const chunks = chunkFiles(repoPath);\n  embed(chunks);\n}"],
  python: ["a.py", "register_route", "class Blueprint:\n    def register_route(self, rule, endpoint):\n        view = make_view(endpoint)\n        self.add_url_rule(rule, view)\n"],
  ruby: ["a.rb", "render_template", "class ActionController\n  def render_template(name, locals)\n    view = lookup_view(name)\n    view.render(locals)\n  end\nend"],
  rust: ["a.rs", "route_request", "struct Router;\nfn route_request(path: &str, method: Method) -> Response {\n    let handler = lookup(path);\n    handler.call(method)\n}"],
  php: ["a.php", "getUserName", "public function getUserName($userId, $default)\n{\n    $user = $this->repo->find($userId);\n    return format_name($user);\n}"],
  elixir: ["a.ex", "match_route", "defmodule Phoenix.Router do\n  def match_route(conn, path) do\n    route = lookup(path)\n    Plug.Conn.assign(conn, :route, route)\n  end\n  defp helper(x), do: x\nend"],
};
let fail = 0;
for (const [lang, [file, want, src]] of Object.entries(S)) {
  const ex = await extractSource(src, null, file);
  const ok = !!ex?.defs.includes(want) && ex.roles.some(([r]) => r === "ROLE_CALL");
  if (!ok) fail++;
  console.log(`${ok ? "PASS" : "FAIL"} ${lang.padEnd(11)} ${ex ? JSON.stringify(ex.roles) : "NULL"}`);
}
process.exit(fail ? 1 : 0);
