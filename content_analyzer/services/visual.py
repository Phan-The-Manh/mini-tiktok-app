"""
visual.py — CLIP visual encoder
===============================

Step 3.5 of the Content Analyzer build.

Encodes the JPEG frames produced by `services/frames.py` into a single
512-dimensional visual vector via `openai/clip-vit-base-patch32`.

Pipeline per video
------------------
    frames (N JPEGs)
        | CLIP image encoder (one forward pass over the whole batch)
        v
    per-frame embeddings (N x 512)
        | L2-normalize each row
        v
    unit-norm per-frame embeddings (N x 512)
        | mean over the frame axis
        v
    visual_embedding (512,)

Why L2-normalize before pooling?
    CLIP image features are not unit-norm. If we averaged them raw,
    frames whose feature happens to have a larger magnitude would
    silently dominate the pooled vector. Per-frame L2 normalization
    gives every frame equal weight in the mean — the standard CLIP
    aggregation recipe.

We do NOT L2-normalize the pooled result here. Step 3.8 (modal fusion)
owns the final normalization across modalities so that the relative
scale of visual vs text is decided in one place.

CPU only
--------
ViT-B/32 inference on 8 frames takes ~200-500 ms on a modern laptop CPU,
which is in line with the TODO target of <60 s end-to-end per video.
GPU is a documented Colab fallback (step 3.11), not part of the local
worker path.

Lifecycle
---------
The model file is ~150 MB and takes a few seconds to load. We pay that
once at worker startup, not once per video:

    encoder = CLIPVisualEncoder()
    encoder.load()                 # explicit load, called by main.py
    vec = encoder.encode(frames)   # per-video, fast path

`get_encoder()` provides a module-level singleton for callers that
prefer lazy initialization (handy in smoke tests).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

import content_analyzer._path  # noqa: F401  side-effect: sys.path + env


log = logging.getLogger(__name__)


# Documented output dimensionality for ViT-B/32. Asserted at load time
# in case a different CLIP variant is configured via CLIP_MODEL.
CLIP_VIT_B32_DIM = 512


class VisualEncodeError(RuntimeError):
    """Raised when no usable visual vector could be produced."""


class CLIPVisualEncoder:
    """Thin wrapper around HuggingFace `CLIPModel` for image-only encoding.

    Lazy: `torch` / `transformers` are imported inside `load()` so that
    other modules in this package can be unit-tested without paying the
    import cost (or even having the packages installed for non-ML
    paths).
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
    ):
        self.model_name = model_name or os.getenv(
            "CLIP_MODEL", "openai/clip-vit-base-patch32"
        )
        self.device = device
        self._model = None
        self._processor = None
        self._torch = None  # cached module handle so encode() doesn't re-import
        self._dim: Optional[int] = None

    # --- Lifecycle -----------------------------------------------------------

    def load(self) -> None:
        """Load CLIP weights + processor. Idempotent."""
        if self._model is not None:
            return

        import torch  # type: ignore[import-not-found]
        from transformers import CLIPModel, CLIPProcessor  # type: ignore[import-not-found]

        log.info("[INFO] loading CLIP model %s on %s", self.model_name, self.device)
        model = CLIPModel.from_pretrained(self.model_name)
        processor = CLIPProcessor.from_pretrained(self.model_name)
        model.to(self.device)
        model.eval()

        self._torch = torch
        self._model = model
        self._processor = processor
        # projection_dim is the image/text feature size CLIP exposes.
        self._dim = int(model.config.projection_dim)

        if self._dim != CLIP_VIT_B32_DIM:
            log.warning(
                "[WARN] CLIP projection_dim=%d (expected %d for ViT-B/32). "
                "Downstream fusion (step 3.8) must accommodate the new dim.",
                self._dim, CLIP_VIT_B32_DIM,
            )
        log.info("[OK] CLIP loaded (projection_dim=%d)", self._dim)

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("encoder not loaded; call .load() first")
        return self._dim

    # --- Encoding ------------------------------------------------------------

    def encode(self, image_paths: list[Path]) -> np.ndarray:
        """Encode a list of frame JPEGs into a single mean-pooled vector.

        Returns a float32 numpy array of shape (embedding_dim,). Raises
        `VisualEncodeError` if zero frames could be opened.
        """
        if not image_paths:
            raise VisualEncodeError("no frames supplied")

        if self._model is None:
            self.load()

        images = self._open_images(image_paths)
        if not images:
            raise VisualEncodeError(
                f"none of the {len(image_paths)} frame(s) could be opened as images"
            )

        torch = self._torch
        processor = self._processor
        model = self._model
        assert torch is not None and processor is not None and model is not None

        inputs = processor(images=images, return_tensors="pt")
        # Move tensors to the configured device.
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Call vision_model + visual_projection directly rather than
            # model.get_image_features(), because some transformers versions
            # return a BaseModelOutputWithPooling wrapper instead of a tensor.
            vision_outputs = model.vision_model(**inputs)
            features = model.visual_projection(vision_outputs.pooler_output)  # (N, dim)

        # L2-normalize per frame, then mean-pool.
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        pooled = features.mean(dim=0)                       # (dim,)

        return pooled.detach().cpu().numpy().astype(np.float32)

    # --- Internals -----------------------------------------------------------

    @staticmethod
    def _open_images(paths: list[Path]) -> list[Image.Image]:
        """Open frame JPEGs as RGB PIL images. Corrupt frames are dropped."""
        out: list[Image.Image] = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                log.warning("[WARN] dropping unreadable frame %s: %s", p, e)
                continue
            out.append(img)
        return out


# --- Module-level singleton --------------------------------------------------

_encoder: Optional[CLIPVisualEncoder] = None


def get_encoder() -> CLIPVisualEncoder:
    """Return a process-wide, lazily-loaded encoder.

    Production startup should instead instantiate the encoder explicitly
    in `main.py` and inject it into the consumer handler — that keeps
    load-time errors at startup, not on the first event delivery.
    """
    global _encoder
    if _encoder is None:
        enc = CLIPVisualEncoder()
        enc.load()
        _encoder = enc
    return _encoder
