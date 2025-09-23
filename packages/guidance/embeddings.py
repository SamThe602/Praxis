"""Lightweight embedding utilities used by the retrieval pipeline.

The production system relies on vector databases such as FAISS.  For the CLI
test environment we provide a simple, deterministic hashing encoder that runs
purely on CPU and has no external dependencies beyond NumPy.  The encoder is
fast enough for unit tests yet stable across Python invocations thanks to
explicit hashing.

The module intentionally exposes just enough surface area for the synthesiser
retrieval flow:

``embed_spec``
    Generate a normalised vector embedding for a structured spec or any object
    that can be serialised to text.

``EmbeddingResult``
    Convenience container bundling the vector together with the tokens that
    were fed into the hasher.  Retrieval consumers primarily use the vector,
    but tests can also introspect the tokens for determinism checks.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "DEFAULT_DIMENSION",
    "EmbeddingResult",
    "embed_spec",
]


DEFAULT_DIMENSION = 128
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(slots=True)
class EmbeddingResult:
    """Wrapper around the embedding vector and its originating tokens."""

    vector: np.ndarray
    tokens: tuple[str, ...]
    raw_text: str

    def as_list(self) -> list[float]:
        """Return the underlying vector as a JSON serialisable list."""

        return self.vector.astype(float).tolist()


def embed_spec(spec: Any, *, dimension: int = DEFAULT_DIMENSION) -> EmbeddingResult:
    """Return a deterministic unit-length embedding for ``spec``.

    The encoder first extracts a flat list of strings from the spec (covering
    identifiers, natural language prompts, constraints, operators, and any
    nested metadata).  Each token is hashed into ``dimension`` buckets using
    BLAKE2b to ensure stability across interpreter runs.  The resulting vector
    is L2-normalised; callers can safely compare embeddings via dot products.

    Parameters
    ----------
    spec:
        Structured specification or arbitrary object describing the task.
    dimension:
        Size of the embedding vector.  The default matches the lightweight
        baseline used throughout the tests; larger dimensions are supported but
        should remain powers of two for best hashing behaviour.
    """

    if dimension <= 0:
        raise ValueError("dimension must be a positive integer")

    tokens = tuple(_extract_tokens(spec))
    vector = _hash_tokens(tokens, dimension)
    raw_text = " ".join(tokens)
    return EmbeddingResult(vector=vector, tokens=tokens, raw_text=raw_text)


def _extract_tokens(spec: Any) -> Iterable[str]:
    """Yield informative tokens for ``spec``.

    The helper attempts to balance determinism with coverage by looking at
    common attributes exposed by :class:`StructuredSpec`.  When the object is a
    mapping or a plain sequence we recurse into its elements.  Ultimately every
    branch is reduced to lower-case word tokens to keep the hashing stable.
    """

    if spec is None:
        return ()
    if isinstance(spec, str):
        return _tokenise(spec)
    if isinstance(spec, Mapping):
        collected: list[str] = []
        for key in sorted(spec.keys()):
            collected.extend(_tokenise(str(key)))
            collected.extend(_extract_tokens(spec[key]))
        return tuple(collected)
    if isinstance(spec, Sequence) and not isinstance(spec, (bytes, bytearray)):
        items: list[str] = []
        for element in spec:
            items.extend(_extract_tokens(element))
        return tuple(items)

    attributes: list[str] = []
    for name in (
        "identifier",
        "natural_prompt",
        "constraints",
        "operators",
        "reuse_pool",
        "goal_tokens",
        "metadata",
        "raw",
    ):
        if hasattr(spec, name):
            attributes.extend(_extract_tokens(getattr(spec, name)))
    if attributes:
        return tuple(attributes)
    return tuple(_tokenise(repr(spec)))


def _tokenise(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in _TOKEN_RE.finditer(text))


def _hash_tokens(tokens: Sequence[str], dimension: int) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "little") % dimension
        vector[index] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector
