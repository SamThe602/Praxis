"""Domain glossary utilities for the translator.

The glossary helps the rule-based translator map noisy natural language tokens
onto canonical task identifiers. Merge behaviour is intentionally predictable:
we treat the canonical identifier as the primary key and later sources override
fields while we union every synonym (deduplicated and lower-cased). This design
keeps deterministic behaviour while still allowing user overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class GlossaryTerm:
    """Representation of a canonical task description."""

    canonical: str
    definition: str
    synonyms: tuple[str, ...] = ()

    @classmethod
    def from_raw(cls, canonical: str, raw: Mapping[str, object]) -> "GlossaryTerm":
        definition = str(raw.get("definition", ""))
        synonyms_raw = raw.get("synonyms", ())
        if isinstance(synonyms_raw, str):
            synonyms_iter: Iterable[str] = (synonyms_raw,)
        elif isinstance(synonyms_raw, Iterable):
            synonyms_iter = (str(item) for item in synonyms_raw)
        else:
            synonyms_iter = ()
        synonyms = tuple(sorted({s.lower().strip() for s in synonyms_iter if s}))
        return cls(canonical=canonical, definition=definition, synonyms=synonyms)


class Glossary:
    """Lookup structure for canonical task terms and their synonyms."""

    def __init__(self, terms: Sequence[GlossaryTerm]):
        self._terms = {term.canonical: term for term in terms}
        keyword_map: Dict[str, str] = {}
        for term in terms:
            keyword_map[term.canonical.lower()] = term.canonical
            for synonym in term.synonyms:
                keyword_map[synonym] = term.canonical
        self._keyword_to_canonical = keyword_map

    def merge(self, other: "Glossary") -> "Glossary":
        terms: Dict[str, GlossaryTerm] = self._terms.copy()
        for canonical, term in other._terms.items():
            if canonical in terms:
                base = terms[canonical]
                merged_synonyms = tuple(sorted({*base.synonyms, *term.synonyms}))
                terms[canonical] = GlossaryTerm(
                    canonical=canonical,
                    definition=term.definition or base.definition,
                    synonyms=merged_synonyms,
                )
            else:
                terms[canonical] = term
        return Glossary(tuple(terms.values()))

    def match(self, text: str) -> str | None:
        """Return the first canonical id with a keyword included in ``text``."""
        lowered = text.lower()
        for keyword, canonical in self._keyword_to_canonical.items():
            if keyword in lowered:
                return canonical
        return None

    def terms(self) -> Sequence[GlossaryTerm]:
        return tuple(self._terms.values())

    def definition_for(self, canonical: str) -> str:
        term = self._terms.get(canonical)
        return term.definition if term else ""


_BASE_TERMS = (
    GlossaryTerm(
        canonical="array_sort",
        definition="Sort a sequence into ascending order.",
        synonyms=("sort", "ascending", "order"),
    ),
    GlossaryTerm(
        canonical="array_reverse",
        definition="Reverse the order of elements in an array.",
        synonyms=("reverse", "reversed", "flip"),
    ),
    GlossaryTerm(
        canonical="graph_path",
        definition="Find a path between two nodes in a graph.",
        synonyms=("graph", "path", "route"),
    ),
    GlossaryTerm(
        canonical="histogram",
        definition="Compute element frequencies for a sequence.",
        synonyms=("histogram", "frequency", "count"),
    ),
    GlossaryTerm(
        canonical="matrix_mult",
        definition="Multiply two matrices.",
        synonyms=("matrix", "product", "multiply"),
    ),
)


def default_glossary() -> Glossary:
    """Return the built-in glossary."""
    return Glossary(_BASE_TERMS)


def build_glossary(overrides: Mapping[str, Mapping[str, object]] | None = None) -> Glossary:
    """Construct glossary merged with optional overrides."""
    glossary = default_glossary()
    if not overrides:
        return glossary
    custom_terms = tuple(
        GlossaryTerm.from_raw(canonical, raw) for canonical, raw in overrides.items()
    )
    return glossary.merge(Glossary(custom_terms))
