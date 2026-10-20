"""
kalpana_rif -- Discrete Relational Inductive Field (pure Python / numpy).

Modules
  vsa_core          HRR tensor algebra: bind (FFT circular convolution), unbind
                    (circular correlation), bundle, normalize, cosine similarity.
  neuro_vocab       NeuroVocab codebook: deterministic *unitary* role vectors
                    (ROLE_CLASS, ROLE_FUNC, ROLE_ARG, ROLE_CALL, ROLE_VAR) plus
                    MiniLM token vectors with isotropic mean-centering.
  neuro_tensorizer  (role, token) pairs -> bundled 384-D state tensor;
                    Python-AST helper for Python sources.
"""
from .neuro_tensorizer import build_tensor, code_to_tensor, extract_ast_pairs
from .neuro_vocab import ROLES, NeuroVocab, make_unitary
from .vsa_core import bind, bundle, cosine_similarity, normalize, unbind

__all__ = [
    "NeuroVocab", "ROLES", "make_unitary",
    "build_tensor", "code_to_tensor", "extract_ast_pairs",
    "bind", "unbind", "bundle", "normalize", "cosine_similarity",
]
__version__ = "0.2.0"
