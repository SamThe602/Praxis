"""Lightweight retrieval layer for Praxis knowledge reuse."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from packages.guidance.embeddings import DEFAULT_DIMENSION, EmbeddingResult, embed_spec

try:  # pragma: no cover - allow retrieval to work without orchestrator package
    from packages.orchestrator.spec_loader import StructuredSpec as _StructuredSpecRuntime
except ImportError:  # pragma: no cover - fallback typing
    _StructuredSpecRuntime = None  # type: ignore

__all__ = [
    "KBEntry",
    "KnowledgeBase",
    "RetrievalConfig",
    "RetrievedSnippet",
    "apply_retrieval",
    "retrieve",
]


KB_ENV_VAR = "PRAXIS_KB_PATH"
KB_FILENAME = "index.json"


@dataclass(slots=True)
class KBEntry:
    spec_id: str
    snippet: str
    embedding: np.ndarray
    metadata: dict[str, Any]

    def as_payload(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "snippet": self.snippet,
            "embedding": self.embedding.astype(float).tolist(),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class RetrievedSnippet:
    spec_id: str
    snippet: str
    score: float
    metadata: dict[str, Any]


@dataclass(slots=True)
class RetrievalConfig:
    top_k: int = 3
    min_score: float = 0.0


class KnowledgeBase:
    """JSON-backed snippet index used by the retrieval pipeline."""

    def __init__(
        self,
        *,
        root: Path,
        entries: Iterable[KBEntry] | None = None,
        dimension: int = DEFAULT_DIMENSION,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = root
        self.path = root / KB_FILENAME
        self.dimension = int(dimension)
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.entries: list[KBEntry] = list(entries or [])
        self._index: dict[tuple[str, str], int] = {}
        for idx, entry in enumerate(self.entries):
            key = self._entry_key(entry.spec_id, _normalise_snippet(entry.snippet))
            self._index[key] = idx

    @classmethod
    def load(cls, root: Path | None = None) -> "KnowledgeBase":
        root = root or _default_root()
        root.mkdir(parents=True, exist_ok=True)
        path = root / KB_FILENAME
        if not path.exists():
            return cls(root=root, entries=(), dimension=DEFAULT_DIMENSION, metadata={})

        raw = json.loads(path.read_text(encoding="utf-8"))
        dimension = int(raw.get("dimension", DEFAULT_DIMENSION))
        metadata = raw.get("metadata", {})
        entries: list[KBEntry] = []
        for item in raw.get("entries", []):
            embedding = np.asarray(item.get("embedding", ()), dtype=np.float32)
            if embedding.size != dimension:
                continue
            spec_id = str(item.get("spec_id", ""))
            snippet = str(item.get("snippet", ""))
            entry_meta = dict(item.get("metadata", {}))
            entries.append(
                KBEntry(
                    spec_id=spec_id,
                    snippet=snippet,
                    embedding=embedding,
                    metadata=entry_meta,
                )
            )
        return cls(root=root, entries=entries, dimension=dimension, metadata=metadata)

    def save(self) -> None:
        payload = {
            "dimension": self.dimension,
            "metadata": self.metadata,
            "entries": [entry.as_payload() for entry in self.entries],
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def record(
        self,
        *,
        spec: Any,
        snippet: str,
        metadata: Mapping[str, Any] | None = None,
        embedding: EmbeddingResult | None = None,
    ) -> KBEntry:
        if not snippet:
            raise ValueError("snippet must be a non-empty string")

        spec_id = _spec_identifier(spec)
        norm_snippet = _normalise_snippet(snippet)
        embedding_result = embedding or embed_spec(spec, dimension=self.dimension)
        vector = np.asarray(embedding_result.vector, dtype=np.float32)
        if vector.size != self.dimension:
            raise ValueError("embedding dimension mismatch")

        entry_meta = dict(metadata or {})
        entry_meta.setdefault("tokens", list(embedding_result.tokens))

        entry = KBEntry(spec_id=spec_id, snippet=snippet, embedding=vector, metadata=entry_meta)
        key = self._entry_key(spec_id, norm_snippet)
        if key in self._index:
            self.entries[self._index[key]] = entry
        else:
            self._index[key] = len(self.entries)
            self.entries.append(entry)
        return entry

    def delete(self, spec_id: str, snippet: str) -> bool:
        key = self._entry_key(spec_id, _normalise_snippet(snippet))
        index = self._index.pop(key, None)
        if index is None:
            return False
        self.entries.pop(index)
        self._reindex()
        return True

    def query(
        self,
        embedding: np.ndarray,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievedSnippet]:
        if not self.entries:
            return []
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.size != self.dimension:
            raise ValueError("embedding dimension mismatch")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return []
        vector = vector / norm

        results: list[RetrievedSnippet] = []
        for entry in self.entries:
            score = float(np.dot(vector, entry.embedding))
            if score < min_score:
                continue
            results.append(
                RetrievedSnippet(
                    spec_id=entry.spec_id,
                    snippet=entry.snippet,
                    score=score,
                    metadata=dict(entry.metadata),
                )
            )

        results.sort(key=lambda item: (-item.score, item.spec_id, item.snippet))
        if top_k > 0:
            return results[:top_k]
        return results

    def merge_payload(self, payload: Mapping[str, Any], *, overwrite: bool = False) -> int:
        dimension = int(payload.get("dimension", self.dimension))
        if dimension != self.dimension:
            self.dimension = dimension

        count = 0
        for item in payload.get("entries", []):
            snippet = str(item.get("snippet", ""))
            spec_id = str(item.get("spec_id", ""))
            embedding = np.asarray(item.get("embedding", ()), dtype=np.float32)
            if not snippet or embedding.size != self.dimension:
                continue
            entry_meta = dict(item.get("metadata", {}))
            entry = KBEntry(
                spec_id=spec_id, snippet=snippet, embedding=embedding, metadata=entry_meta
            )
            key = self._entry_key(spec_id, _normalise_snippet(snippet))
            if key in self._index and not overwrite:
                continue
            if key in self._index:
                self.entries[self._index[key]] = entry
            else:
                self._index[key] = len(self.entries)
                self.entries.append(entry)
            count += 1
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            self.metadata.update(metadata)
        return count

    def _entry_key(self, spec_id: str, snippet: str) -> tuple[str, str]:
        return (spec_id, snippet)

    def _reindex(self) -> None:
        self._index = {}
        for idx, entry in enumerate(self.entries):
            key = self._entry_key(entry.spec_id, _normalise_snippet(entry.snippet))
            self._index[key] = idx


def retrieve(
    spec: Any,
    *,
    config: RetrievalConfig | None = None,
    kb: KnowledgeBase | None = None,
) -> tuple[RetrievedSnippet, ...]:
    config = config or RetrievalConfig()
    kb = kb or KnowledgeBase.load()
    embedding = embed_spec(spec, dimension=kb.dimension)
    candidates = kb.query(
        embedding.vector,
        top_k=config.top_k,
        min_score=config.min_score,
    )
    return tuple(candidates)


def apply_retrieval(
    spec: Any,
    *,
    config: RetrievalConfig | None = None,
    kb: KnowledgeBase | None = None,
) -> Any:
    config = config or RetrievalConfig()
    kb = kb or KnowledgeBase.load()
    candidates = retrieve(spec, config=config, kb=kb)
    if not candidates:
        return spec

    existing = tuple(getattr(spec, "reuse_pool", ()))
    retrieved_snippets = tuple(candidate.snippet for candidate in candidates)
    reuse_pool = _merge_reuse(existing, retrieved_snippets)

    metadata = getattr(spec, "metadata", {})
    metadata_copy = dict(metadata) if isinstance(metadata, Mapping) else {}
    metadata_copy["retrieval"] = {
        "top_k": config.top_k,
        "min_score": config.min_score,
        "results": [
            {
                "spec_id": candidate.spec_id,
                "snippet": candidate.snippet,
                "score": candidate.score,
                "metadata": candidate.metadata,
            }
            for candidate in candidates
        ],
    }

    if _StructuredSpecRuntime is not None:
        try:
            if isinstance(spec, _StructuredSpecRuntime):
                return replace(spec, reuse_pool=reuse_pool, metadata=metadata_copy)
        except TypeError:
            pass
    if isinstance(spec, dict):
        spec = dict(spec)
        spec["reuse_pool"] = reuse_pool
        spec.setdefault("metadata", metadata_copy)
        return spec
    setattr(spec, "reuse_pool", reuse_pool)
    setattr(spec, "metadata", metadata_copy)
    return spec


def _merge_reuse(existing: Sequence[str], retrieved: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []
    for snippet in (*retrieved, *existing):
        normalised = _normalise_snippet(snippet)
        if normalised in seen:
            continue
        seen.add(normalised)
        merged.append(snippet)
    return tuple(merged)


def _spec_identifier(spec: Any) -> str:
    if hasattr(spec, "identifier"):
        return str(getattr(spec, "identifier"))
    if isinstance(spec, Mapping) and "id" in spec:
        return str(spec["id"])
    text = getattr(spec, "natural_prompt", None)
    if isinstance(text, str) and text:
        return text[:64]
    return repr(spec)[:64]


def _normalise_snippet(snippet: str) -> str:
    return " ".join(snippet.strip().split())


def _default_root() -> Path:
    override = os.environ.get(KB_ENV_VAR)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data" / "processed" / "kb"
