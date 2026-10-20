from __future__ import annotations
"""
vsa_core.py -- Holographic Reduced Representations (HRR) math core.

Pure numpy. All vectors are real-valued 1-D arrays of length D (default 10,000).

Key facts that make HRR work:
  * Random Gaussian vectors with variance 1/D have L2 norm ~= 1 and are
    quasi-orthogonal in high dimension: cos(a, b) ~ N(0, 1/D), i.e. |cos| ~ 0.01
    for D = 10,000.
  * Circular convolution (bind) produces a vector dissimilar to BOTH inputs,
    so "role * value" pairs do not leak into each other.
  * Circular correlation (unbind) approximately inverts convolution.
"""

import numpy as np

DEFAULT_DIM = 10_000


def generate_hypervector(dim: int = DEFAULT_DIM, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a random hypervector with elements ~ N(0, 1/dim).

    Mean 0, variance 1/dim => expected L2 norm of 1. In high-D space any two
    such vectors are nearly orthogonal, which is the foundation of VSA.
    """
    rng = rng if rng is not None else np.random.default_rng()
    return rng.normal(loc=0.0, scale=1.0 / np.sqrt(dim), size=dim)


def bind(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Circular convolution  c = v1 (*) v2.

    Direct convolution is O(D^2). By the convolution theorem it becomes an
    element-wise product in the frequency domain, giving O(D log D):

        c = IFFT( FFT(v1) . FFT(v2) )

    We take the real part because inputs are real (imag part is ~1e-17 noise).
    """
    return np.real(np.fft.ifft(np.fft.fft(v1) * np.fft.fft(v2)))


def unbind(bound_tensor: np.ndarray, v_role: np.ndarray) -> np.ndarray:
    """Circular correlation  v ~= role (#) bound_tensor.

    Correlation is convolution with the *involution* of the role vector. In the
    frequency domain the involution is simply the complex conjugate:

        v ~= IFFT( conj(FFT(role)) . FFT(bound) )

    Because |FFT(role)|^2 ~= 1 on average for our random vectors, this recovers
    a noisy copy of the value bound to `role`; all other bundled pairs turn
    into quasi-orthogonal noise.
    """
    return np.real(np.fft.ifft(np.conj(np.fft.fft(v_role)) * np.fft.fft(bound_tensor)))


def normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector (safe against zero norm)."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def bundle(list_of_vectors: list[np.ndarray]) -> np.ndarray:
    """Superposition: element-wise sum followed by L2 normalization.

    The result stays similar to every constituent (cos ~ 1/sqrt(k) for k
    orthogonal items), so it acts like a set/record held in a single tensor.
    """
    return normalize(np.sum(np.stack(list_of_vectors), axis=0))


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Dot product of the two L2-normalized vectors (normalizes defensively)."""
    return float(np.dot(normalize(v1), normalize(v2)))
