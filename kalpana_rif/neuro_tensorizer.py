"""
neuro_tensorizer.py -- (role, token) pairs -> neuro-symbolic HRR tensor (D = 384).

  build_tensor(pairs, vocab)       language-agnostic core API
  extract_ast_pairs(source)        Python-specific helper (stdlib `ast`)
  code_to_tensor(source, vocab)    Python convenience wrapper
"""
from __future__ import annotations

import ast

import numpy as np

from .neuro_vocab import NeuroVocab
from .vsa_core import bind, bundle


def extract_ast_pairs(source_code: str) -> list[tuple[str, str]]:
    """Walk the AST and return structural (role, identifier) pairs.

    ast.walk is breadth-first; order is irrelevant since bundling commutes.
    """
    tree = ast.parse(source_code)
    pairs: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            pairs.append(("ROLE_CLASS", node.name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pairs.append(("ROLE_FUNC", node.name))
        elif isinstance(node, ast.arg):
            pairs.append(("ROLE_ARG", node.arg))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            # Plain calls like foo() only; obj.foo() is skipped.
            pairs.append(("ROLE_CALL", node.func.id))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            # Assignment targets only (x = ...), not reads.
            pairs.append(("ROLE_VAR", node.id))
    return pairs


def build_tensor(pairs: list[tuple[str, str]], vocab: NeuroVocab) -> np.ndarray:
    """Language-agnostic core: bundle(bind(role, token)) over pre-extracted pairs.

    `pairs` may come from any parser (Python ast, Tree-sitter for Go/JS, ...).
    Roles must be ones known to `vocab` (see neuro_vocab.ROLES); tokens can be
    any string.
    """
    if not pairs:
        raise ValueError("No structural pairs provided.")
    unknown = {r for r, _ in pairs if r not in vocab._roles}
    if unknown:
        raise ValueError(f"Unknown role(s) {sorted(unknown)}; "
                         f"supported: {sorted(vocab._roles)}")
    return bundle([bind(vocab.get_role(r), vocab.get_token(t)) for r, t in pairs])


def code_to_tensor(source_code: str, vocab: NeuroVocab) -> np.ndarray:
    """Python convenience wrapper (backward compatible)."""
    return build_tensor(extract_ast_pairs(source_code), vocab)


__all__ = ["build_tensor", "code_to_tensor", "extract_ast_pairs"]
