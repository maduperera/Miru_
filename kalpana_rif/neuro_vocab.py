"""
neuro_vocab.py -- Neuro-symbolic codebook (D = 384).

  * ROLES  -> random hypervectors (symbolic, quasi-orthogonal)
  * TOKENS -> MiniLM (all-MiniLM-L6-v2) sentence embeddings (semantic)

Both live in the same 384-D space so they can be bound with FFT convolution.

Design note -- unitary roles:
  At D=10,000 random Gaussian roles are "almost" unitary (|FFT(r)_k| ~ 1).
  At D=384 the spectrum magnitude fluctuates a lot, which distorts the
  token geometry after binding. We therefore make role vectors *unitary*
  (every FFT coefficient has magnitude 1). Then binding is an isometry:

      cos(bind(R, a), bind(R, b)) == cos(a, b)        (exactly)

  so MiniLM synonym similarity survives binding perfectly within a role,
  while different roles remain quasi-orthogonal (noise ~ 1/sqrt(384) ~ 0.05).
"""
from __future__ import annotations

import hashlib

import numpy as np

from .vsa_core import generate_hypervector, normalize

MINILM_NAME = "all-MiniLM-L6-v2"
MINILM_DIM = 384
ROLES = ("ROLE_CLASS", "ROLE_FUNC", "ROLE_ARG", "ROLE_CALL", "ROLE_VAR")

# Default reference vocabulary for the isotropic (mean-centering) transform.
# MiniLM embeddings of short words sit in a narrow cone: unrelated words share
# a large common component, giving a positive cosine baseline. Subtracting the
# mean over a representative programming vocabulary removes that component.
#
# SANITIZED: contains no word (or close stem) used by the benchmark snippets
# (fetch, storage/store, data, retrieve, class, def, self), so benchmarks are
# zero-shot w.r.t. the baseline. In production prefer `precomputed_mean`
# derived from the target corpus, or pass `reference_words`.
REFERENCE_WORDS = (
    "var", "string", "loop", "get", "set", "node", "server", "client",
    "request", "response", "list", "dict", "array", "map", "filter", "reduce",
    "index", "key", "value", "item", "user", "file", "path", "read", "write",
    "open", "close", "load", "save", "cache", "query", "search", "find",
    "update", "delete", "create", "insert", "parse", "render", "config",
    "error", "exception", "log", "print", "return", "result", "count", "size",
    "length", "buffer", "stream", "socket", "thread", "lock", "queue", "stack",
    "tree", "graph", "edge", "vertex", "model", "view", "controller",
    "handler", "event", "callback", "token", "session", "auth", "password",
    "database", "table", "row", "column", "schema", "import", "module",
    "package", "test", "assert", "init", "args", "name", "type", "int",
    "float", "bool", "true", "false", "none", "start", "stop", "integer",
    "boolean", "system", "network", "route", "timer", "format", "encode",
)

# Words that must never appear in the default list (benchmark targets).
_BENCHMARK_WORDS = frozenset(
    {"fetch", "storage", "store", "data", "retrieve", "class", "def", "self"})
assert not _BENCHMARK_WORDS & set(REFERENCE_WORDS), "benchmark leak in REFERENCE_WORDS"


def make_unitary(v: np.ndarray) -> np.ndarray:
    """Project a real vector onto the unitary set: keep FFT phases, set |.|=1.

    Using the real-FFT keeps conjugate symmetry, so the result is real.
    The resulting vector also has L2 norm exactly 1.
    """
    f = np.fft.rfft(v)
    f = f / np.maximum(np.abs(f), 1e-12)
    return np.fft.irfft(f, n=len(v))


class NeuroVocab:
    """Symbolic roles + neural (MiniLM) tokens in one 384-D VSA space."""

    def __init__(self, seed: int = 42, model_name: str = MINILM_NAME,
                 center: bool = True, model=None,
                 reference_words=None, precomputed_mean=None):
        """
        center:           apply the isotropic mean-centering transform.
        model:            optional pre-loaded SentenceTransformer.
        reference_words:  custom word list to compute the mean from.
        precomputed_mean: 384-D mean vector (e.g. from a production corpus);
                          takes priority over reference_words.

        Mean resolution order:
            precomputed_mean > reference_words > default REFERENCE_WORDS
        """
        if model is None:
            # Imported lazily so the rest of the project stays numpy-only.
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device="cpu")
        self.model = model
        self.dim = self.model.get_sentence_embedding_dimension()
        assert self.dim == MINILM_DIM, f"expected 384-D model, got {self.dim}"
        self.seed = seed
        self._roles = {r: self._make_role(r) for r in ROLES}
        self._tokens: dict[str, np.ndarray] = {}

        # Isotropic transform. Zero vector => centering disabled.
        self.center = center
        if not center:
            self.mean_vector, self.mean_source = np.zeros(self.dim), "none"
        elif precomputed_mean is not None:
            mean = np.asarray(precomputed_mean, dtype=np.float64).reshape(-1)
            if mean.shape != (self.dim,):
                raise ValueError(f"precomputed_mean must be {self.dim}-D, got {mean.shape}")
            self.mean_vector, self.mean_source = mean, "precomputed"
        else:
            words = list(reference_words) if reference_words is not None else list(REFERENCE_WORDS)
            if not words:
                raise ValueError("reference_words must not be empty")
            self.mean_vector = self.compute_mean(words)
            self.mean_source = "custom" if reference_words is not None else "default"

    def compute_mean(self, words) -> np.ndarray:
        """Mean raw MiniLM embedding over `words`.

        Use this on a production corpus' identifiers, save the result
        (np.save), and pass it back later as `precomputed_mean`.
        """
        emb = self.model.encode(list(words), convert_to_numpy=True, batch_size=256)
        return emb.astype(np.float64).mean(axis=0)

    def _make_role(self, name: str) -> np.ndarray:
        """Deterministic random unitary role vector (stable across runs)."""
        digest = hashlib.sha256(f"{self.seed}:{name}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        return make_unitary(generate_hypervector(self.dim, rng))

    def get_role(self, name: str) -> np.ndarray:
        return self._roles[name]

    def get_token(self, token: str) -> np.ndarray:
        """normalize(MiniLM(token) - mean_vector), cached."""
        if token not in self._tokens:
            emb = self.model.encode(token, convert_to_numpy=True).astype(np.float64)
            self._tokens[token] = normalize(emb - self.mean_vector)
        return self._tokens[token]

    def __getitem__(self, key: str) -> np.ndarray:
        """vocab['ROLE_FUNC'] -> role vector, vocab['fetch'] -> token vector."""
        return self.get_role(key) if key in self._roles else self.get_token(key)
