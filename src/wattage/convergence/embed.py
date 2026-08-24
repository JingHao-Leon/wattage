"""Embedding backend (doc §5.2/§8.1) behind an interface so `--embed api` can
swap in later without touching detector logic.

The doc's own principles are in tension here: §5.2 calls for
"sentence-transformers, small offline model by default", but §3.5/§8.1
insist the base install stays fast, dependency-light, and fully functional
offline — and the convergence engine is the flagship feature, meant to work
zero-config (§10). Requiring a ~100MB+ optional ML dependency just to get a
default-mode result would contradict that. The resolution: "local" tries
sentence-transformers if `wattage[embeddings]` is installed (best quality),
and falls back to a dependency-free hashing embedder — a real, if cruder,
semantic-similarity proxy, not a stub — if it isn't. Either way the
convergence engine works out of the box; quality just improves with the
extra installed.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from abc import ABC, abstractmethod
from typing import Any

_NGRAM_SIZE = 4
_VECTOR_DIM = 256


class Embedder(ABC):
    @abstractmethod
    def similarity(self, text_a: str, text_b: str) -> float:
        """Similarity in [0, 1]; 1.0 = identical/near-duplicate, 0.0 = unrelated."""

    def novelty(self, text: str, priors: list[str]) -> float:
        """1 - max similarity vs any prior text. No text, or no non-empty
        priors, yields a neutral 0.5 — no signal either way, never a
        fabricated confident 0 or 1."""
        if not text:
            return 0.5
        non_empty_priors = [p for p in priors if p]
        if not non_empty_priors:
            return 0.5
        return 1.0 - max(self.similarity(text, prior) for prior in non_empty_priors)


class NullEmbedder(Embedder):
    """--embed off: no semantic signal at all, always neutral."""

    def similarity(self, text_a: str, text_b: str) -> float:
        return 0.5

    def novelty(self, text: str, priors: list[str]) -> float:
        return 0.5


def _shingles(text: str, n: int = _NGRAM_SIZE) -> list[str]:
    normalized = " ".join(text.lower().split())
    if len(normalized) < n:
        return [normalized] if normalized else []
    return [normalized[i : i + n] for i in range(len(normalized) - n + 1)]


def _hash_vector(text: str, dim: int = _VECTOR_DIM) -> list[float]:
    vector = [0.0] * dim
    for shingle in _shingles(text):
        digest = hashlib.sha256(shingle.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class HashEmbedder(Embedder):
    """Feature-hashed character n-grams into a fixed-size signed vector,
    compared by cosine similarity. Deterministic, no dependencies, no
    network — texts sharing substrings land in overlapping buckets, a crude
    but real (not fabricated) novelty signal.

    Vectors are memoized per instance: novelty computations compare each
    iteration against every prior one (O(n²) pairs), and re-deriving the
    same text's vector for every pair made long production loops painfully
    slow for no accuracy gain.
    """

    def __init__(self) -> None:
        self._vector_cache: dict[str, list[float]] = {}

    def _vector(self, text: str) -> list[float]:
        cached = self._vector_cache.get(text)
        if cached is None:
            cached = _hash_vector(text)
            self._vector_cache[text] = cached
        return cached

    def similarity(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.5
        cosine = _cosine(self._vector(text_a), self._vector(text_b))
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


class SentenceTransformerEmbedder(Embedder):
    """Wraps the optional `wattage[embeddings]` extra for materially better
    novelty detection on genuinely different phrasings of the same idea.
    Falls back to HashEmbedder (with a one-time warning) if the extra isn't
    installed, rather than crashing — this is what "local" resolves to."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Any | None = None
        self._fallback: Embedder | None = None
        self._vector_cache: dict[str, Any] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._fallback is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            # Model construction can fail for reasons well beyond a missing
            # package: no network for the first-run download, a blocked HF
            # cache dir, a corrupted cache. Every one of those must degrade
            # to the hash embedder — a CI gate that hard-crashes on an
            # offline runner is worse than a cruder similarity signal.
            self._model = SentenceTransformer(self._model_name)
        except ImportError:
            warnings.warn(
                "sentence-transformers isn't installed (install 'wattage[embeddings]', or "
                "with uvx: uvx --from 'wattage[embeddings]' wattage); falling back to the "
                "built-in hash embedder for convergence signals.",
                stacklevel=3,
            )
            self._fallback = HashEmbedder()
        except Exception as exc:  # noqa: BLE001 — any load failure degrades, never crashes
            warnings.warn(
                f"sentence-transformers model {self._model_name!r} could not be loaded "
                f"({exc.__class__.__name__}: {exc}); falling back to the built-in hash "
                "embedder for convergence signals.",
                stacklevel=3,
            )
            self._fallback = HashEmbedder()

    def _vector(self, text: str) -> Any:
        cached = self._vector_cache.get(text)
        if cached is None:
            assert self._model is not None
            cached = self._model.encode([text], normalize_embeddings=True)[0]
            self._vector_cache[text] = cached
        return cached

    def similarity(self, text_a: str, text_b: str) -> float:
        self._ensure_loaded()
        if self._fallback is not None:
            return self._fallback.similarity(text_a, text_b)
        if not text_a or not text_b:
            return 0.5
        cosine = float(self._vector(text_a) @ self._vector(text_b))
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def build_embedder(mode: str) -> Embedder:
    if mode == "off":
        return NullEmbedder()
    if mode == "local":
        return SentenceTransformerEmbedder()
    if mode == "api":
        raise NotImplementedError(
            "--embed api is a documented future extension point (doc §5.2); "
            "no external embedding API is integrated yet."
        )
    raise ValueError(f"Unknown embed mode: {mode!r}")
