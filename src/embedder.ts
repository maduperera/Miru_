/**
 * Query/document embedder used to build the Miru index and to embed benchmark queries.
 *
 *  - `ds1`     Takara DS1 over the OpenAI-compatible endpoint (what Miru itself calls).
 *              Requires TAKARA_API_KEY. Default model `ds1-miru-int8` (256-D).
 *  - `minilm`  Local all-MiniLM-L6-v2 truncated to 256-D (src/minilm.ts). Default when no key is set.
 *
 * Selection: EMBEDDER=ds1|minilm. Default = ds1 when TAKARA_API_KEY is set, else minilm.
 * The index and the queries MUST use the same model — rebuild the index when switching.
 */
import { MINILM_DIM, MINILM_MODEL_NAME, minilmEmbedBatch } from "./minilm.ts";

export interface Embedder {
  readonly name: string;
  readonly dim: number;
  embedQuery(text: string): Promise<Float32Array>;
  /** Batch embedding (used by the local OpenAI-compatible endpoint that Miru indexes through). */
  embedBatch(texts: string[]): Promise<Float32Array[]>;
}

export function createMinilmEmbedder(): Embedder {
  return {
    name: MINILM_MODEL_NAME,
    dim: MINILM_DIM,
    embedQuery: async (t) => (await minilmEmbedBatch([t]))[0],
    embedBatch: minilmEmbedBatch,
  };
}

export function createDs1Embedder(opts?: { baseUrl?: string; model?: string; apiKey?: string }): Embedder {
  const baseUrl = (opts?.baseUrl ?? process.env.MIRU_OPENAI_BASE_URL ?? "https://infer.takara.ai/v1").replace(/\/$/, "");
  const model = opts?.model ?? process.env.MIRU_OPENAI_EMBEDDING_MODEL ?? "ds1-miru-int8";
  const apiKey = opts?.apiKey ?? process.env.TAKARA_API_KEY;
  if (!apiKey) throw new Error("TAKARA_API_KEY is required for the DS1 embedder");
  const call = async (input: string | string[], query: boolean): Promise<Float32Array[]> => {
    const res = await fetch(`${baseUrl}/embeddings`, {
      method: "POST",
      headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json", ...(query ? { "X-Q": "1" } : {}) },
      body: JSON.stringify({ model, input }),
    });
    if (!res.ok) throw new Error(`DS1 embedding error ${res.status}: ${(await res.text()).slice(0, 300)}`);
    const payload = (await res.json()) as { data: { embedding: number[] | { values: number[]; scale: number; zero_point: number } }[] };
    return payload.data.map(({ embedding: e }) =>
      Array.isArray(e) ? new Float32Array(e) : Float32Array.from(e.values, (v) => (v - e.zero_point) * e.scale),
    );
  };
  return { name: model, dim: 256, embedQuery: async (t) => (await call(t, true))[0], embedBatch: (ts) => call(ts, false) };
}

export function embedderChoice(): "ds1" | "minilm" {
  const c = process.env.EMBEDDER ?? (process.env.TAKARA_API_KEY ? "ds1" : "minilm");
  if (c !== "ds1" && c !== "minilm") throw new Error(`unknown EMBEDDER=${c} (expected ds1 or minilm)`);
  return c;
}

export function createEmbedder(): Embedder {
  return embedderChoice() === "ds1" ? createDs1Embedder() : createMinilmEmbedder();
}
