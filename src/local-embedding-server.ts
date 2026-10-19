/**
 * OpenAI-compatible /v1/embeddings server backed by a LOCAL embedder (default: MiniLM truncated to 256-D;
 * see src/embedder.ts). Lets the real, unmodified `@takara-ai/miru-code` index a repo with no API key:
 *
 *   MIRU_OPENAI_BASE_URL=http://127.0.0.1:8787/v1 TAKARA_API_KEY=local \
 *   MIRU_OPENAI_EMBEDDING_MODEL=<embedder.name> MIRU_EMBEDDING_DIMENSIONS=256 ...
 *
 * Swapping in real DS1 later = drop those env overrides and set a real TAKARA_API_KEY.
 */
import http from "node:http";
import { createEmbedder, type Embedder } from "./embedder.ts";

export function startEmbeddingServer(embedder: Embedder, port = 8787): Promise<http.Server> {
  // Progress counter: texts embedded so far (logged every ~100 so a long run's progress is observable).
  let embedded = 0;
  let lastLogged = 0;
  const t0 = Date.now();
  const server = http.createServer((req, res) => {
    if (req.method !== "POST" || !req.url?.endsWith("/embeddings")) {
      res.writeHead(404).end("not found");
      return;
    }
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", async () => {
      try {
        const body = JSON.parse(Buffer.concat(chunks).toString("utf-8")) as { input: string | string[] };
        const inputs = (Array.isArray(body.input) ? body.input : [body.input]).map(String);
        const vecs = await embedder.embedBatch(inputs);
        embedded += inputs.length;
        if (embedded - lastLogged >= 100) {
          lastLogged = embedded;
          const mins = (Date.now() - t0) / 60000;
          console.log(`[embed] ${embedded} texts in ${mins.toFixed(1)} min (${(embedded / mins).toFixed(0)}/min) ${new Date().toISOString()}`);
        }
        const data = vecs.map((v, index) => ({ object: "embedding", index, embedding: Array.from(v) }));
        res.writeHead(200, { "Content-Type": "application/json", Connection: "close" });
        res.end(JSON.stringify({ object: "list", data, model: embedder.name, usage: { dim: embedder.dim } }));
      } catch (e) {
        // 500 is in Miru's transient-retry set (400 would abort indexing).
        res.writeHead(500, { Connection: "close" }).end(String(e));
      }
    });
  });
  // Inference is serial and slow: a client reusing a keep-alive socket that Node just timed out gets ECONNRESET,
  // so every response closes its connection, and no server-side request timeout applies.
  server.keepAliveTimeout = 0;
  server.requestTimeout = 0;
  server.headersTimeout = 0;
  return new Promise((resolve) => server.listen(port, "127.0.0.1", () => resolve(server)));
}

// Run standalone: `npm run embed-server`
if (import.meta.url === `file://${process.argv[1]}`) {
  const port = Number(process.env.EMBED_PORT ?? 8787);
  const e = createEmbedder();
  startEmbeddingServer(e, port).then(() => console.log(`${e.name} embedding server on http://127.0.0.1:${port}/v1 (dim=${e.dim})`));
}
