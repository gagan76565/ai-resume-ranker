"""Optional semantic matching with MiniLM-L6-v2 run through ONNX Runtime (no PyTorch).

* Loaded **once per process** (lazy, thread-safe) from ``SEMANTIC_MODEL_DIR``
  (default ``models/all-MiniLM-L6-v2``; download with ``python scripts/fetch_model.py``).
* ``SEMANTIC_ENABLED``: ``auto`` (default — use the model if its files are present), ``1`` or ``0``.
* Embeddings are cached in memory by text hash (bounded LRU). Nothing is written to disk.
* If the model is missing, disabled or fails to load, :func:`get_encoder` returns ``None`` and the
  scoring engine falls back to lexical evidence and says so in the response. There is no external API.

Similarity → score mapping (heuristic, documented in docs/ai/SCORING_METHODOLOGY.md): cosine similarities
of MiniLM sentence embeddings are rescaled with ``(cos − 0.15) / 0.50`` and clipped to [0, 1], so ~0.15 or
less counts as unrelated and ≥ 0.65 as a strong paraphrase match.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "models" / "all-MiniLM-L6-v2"
MAX_TOKENS = 128
LOW, HIGH = 0.15, 0.65

_lock = threading.Lock()
_state: dict = {"encoder": None, "loaded": False, "error": None}


class Encoder:
    def __init__(self, model_dir: Path, cache_size: int = 4096):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.environ.get("SEMANTIC_THREADS", "1"))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")  # noqa: S106 - tokenizer padding symbol, not a secret
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_size = cache_size
        self.name = "all-MiniLM-L6-v2 (ONNX int8)"

    def _run(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feeds)[0]  # (batch, tokens, 384)
        weights = mask[..., None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        return pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]  # noqa: S324 - cache key only
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        for start in range(0, len(missing), batch_size):
            chunk = missing[start:start + batch_size]
            vectors = self._run([texts[i] for i in chunk])
            for i, vector in zip(chunk, vectors, strict=True):
                self.cache[keys[i]] = vector.astype(np.float32)
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        for k in keys:
            self.cache.move_to_end(k)
        return np.stack([self.cache[k] for k in keys]) if keys else np.zeros((0, 384), dtype=np.float32)


def get_encoder(env=os.environ) -> Encoder | None:
    """The shared encoder, or None when semantic matching is disabled or unavailable."""
    setting = str(env.get("SEMANTIC_ENABLED", "auto")).strip().lower()
    if setting in {"0", "false", "no", "off"}:
        return None
    if _state["loaded"]:
        return _state["encoder"]
    with _lock:
        if not _state["loaded"]:
            model_dir = Path(env.get("SEMANTIC_MODEL_DIR") or DEFAULT_DIR)
            try:
                if not (model_dir / "model.onnx").exists():
                    raise FileNotFoundError(f"no model at {model_dir}")
                _state["encoder"] = Encoder(model_dir)
                log.info("Semantic matching enabled (%s)", _state["encoder"].name)
            except Exception as exc:  # noqa: BLE001 - any failure means lexical fallback
                _state["error"] = str(exc)
                _state["encoder"] = None
                log.warning("Semantic matching unavailable, using lexical fallback: %s", exc)
            _state["loaded"] = True
    return _state["encoder"]


def status() -> dict:
    return {"enabled": _state["encoder"] is not None, "loaded": _state["loaded"],
            "model": _state["encoder"].name if _state["encoder"] else None, "error": _state["error"]}


def reset_for_tests() -> None:
    with _lock:
        _state.update({"encoder": None, "loaded": False, "error": None})


def rescale(cosine: float) -> float:
    return float(min(1.0, max(0.0, (cosine - LOW) / (HIGH - LOW))))


_SENTENCE = re.compile(r"(?<=[.!?;])\s+|\n")


def sentences(text: str, start: int, end: int, min_chars: int = 25) -> list[tuple[str, int, int]]:
    """Split text[start:end] into sentences with spans; very short fragments are dropped."""
    out, cursor = [], start
    for piece in _SENTENCE.split(text[start:end]):
        idx = text.find(piece, cursor, end)
        if idx < 0:
            continue
        cursor = idx + len(piece)
        if len(piece.strip()) >= min_chars:
            out.append((piece.strip(), idx, idx + len(piece)))
    return out


def match(encoder: Encoder, queries: list[str], candidates: list[tuple[str, int, int, str]], max_candidates: int = 80) -> list[dict]:
    """For each query (JD requirement sentence) return the best resume sentence.

    ``candidates`` are (sentence, start, end, section). Returns per query:
    ``{"query", "score" (rescaled 0..1), "cosine", "best": {"text", "start", "end", "section"} | None}``.
    """
    if not queries:
        return []
    candidates = candidates[:max_candidates]
    if not candidates:
        return [{"query": q, "score": 0.0, "cosine": 0.0, "best": None} for q in queries]
    q_vec = encoder.encode(queries)
    c_vec = encoder.encode([c[0] for c in candidates])
    sims = q_vec @ c_vec.T
    out = []
    for qi, query in enumerate(queries):
        j = int(np.argmax(sims[qi]))
        cos = float(sims[qi, j])
        best = candidates[j]
        out.append({"query": query, "score": round(rescale(cos), 4), "cosine": round(cos, 4),
                    "best": {"text": best[0], "start": best[1], "end": best[2], "section": best[3]}})
    return out
