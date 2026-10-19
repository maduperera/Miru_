/**
 * Local open-source embedder, used when no TAKARA_API_KEY is available (no network after first download).
 *
 *   model : Xenova/all-MiniLM-L6-v2 (transformers.js / ONNX)
 *   dim   : 384 → first 256 dims → L2-re-normalised, to match the shape of Miru's `ds1-miru-int8` (256-D,
 *           unit-norm). MiniLM is not a Matryoshka model, so truncation loses some quality. It is NOT DS1.
 */
import { pipeline, type FeatureExtractionPipeline } from "@xenova/transformers";

export const MINILM_MODEL_NAME = "minilm-l6-256";
export const MINILM_DIM = 256;
const HF_MODEL = "Xenova/all-MiniLM-L6-v2";
const SUB_BATCH = 16;

let extractor: Promise<FeatureExtractionPipeline> | null = null;
const getExtractor = () => (extractor ??= pipeline("feature-extraction", HF_MODEL) as Promise<FeatureExtractionPipeline>);

/** Serialise ONNX calls (Miru fires concurrent HTTP requests; the session is single-threaded anyway). */
let chain: Promise<unknown> = Promise.resolve();

export function truncateNormalize(full: Float32Array, dim = MINILM_DIM): Float32Array {
  const out = full.slice(0, dim);
  let n = 0;
  for (let i = 0; i < out.length; i++) n += out[i] * out[i];
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < out.length; i++) out[i] /= n;
  return out;
}

export function minilmEmbedBatch(texts: string[]): Promise<Float32Array[]> {
  const run = async (): Promise<Float32Array[]> => {
    const ex = await getExtractor();
    const result: Float32Array[] = [];
    for (let i = 0; i < texts.length; i += SUB_BATCH) {
      const batch = texts.slice(i, i + SUB_BATCH).map((t) => (t.trim() ? t : " "));
      const out = await ex(batch, { pooling: "mean", normalize: true });
      const [rows, d] = out.dims as [number, number];
      const data = out.data as Float32Array;
      for (let r = 0; r < rows; r++) result.push(truncateNormalize(data.slice(r * d, (r + 1) * d)));
    }
    return result;
  };
  const p = chain.then(run, run);
  chain = p.catch(() => undefined);
  return p;
}
