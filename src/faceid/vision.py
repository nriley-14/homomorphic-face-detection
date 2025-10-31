"""Computer vision utilities for face detection and embedding projection.

This module provides pure functional operations for face detection, embedding
normalization, projection, and visualization using InsightFace and OpenCV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple
import numpy as np
import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis

from faceid.config import EMB_DIM, PROJ_DIM

ort.preload_dlls() # Load GPU dlls

__all__ = [
    "l2_normalize",
    "ensure_face_app",
    "pick_largest",
    "draw_bbox_and_label",
    "load_projection",
    "get_or_make_projection",
    "compute_projected_embedding",
]


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector with numerical stability.

    Args:
        v: Input vector of any shape.

    Returns:
        L2-normalized vector with same shape.
    """
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def ensure_face_app(
    device: str,
    det_size: Tuple[int, int] = (512, 512),
    name: str = "buffalo_s",
) -> FaceAnalysis:
    """Create InsightFace app with device selection.

    Args:
        device: Device mode - 'gpu' (require CUDA), 'cpu' (force CPU), or 'auto'.
        det_size: Detection input size as (width, height).
        name: InsightFace model pack name.

    Returns:
        Initialized FaceAnalysis app.

    Raises:
        RuntimeError: If GPU requested but CUDA unavailable.
    """
    providers = ort.get_available_providers()
    has_cuda = "CUDAExecutionProvider" in providers

    if device == "gpu":
        if not has_cuda:
            raise RuntimeError(
                "CUDAExecutionProvider unavailable but --device gpu requested."
            )
        use_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
        print("[VISION] Using GPU (CUDAExecutionProvider).")
    elif device == "cpu":
        use_providers = ["CPUExecutionProvider"]
        ctx_id = -1
        print("[VISION] Using CPU (CPUExecutionProvider).")
    else:
        use_providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if has_cuda
            else ["CPUExecutionProvider"]
        )
        ctx_id = 0 if has_cuda else -1
        print(f"[VISION] Using {'GPU' if has_cuda else 'CPU'} (auto).")

    app = FaceAnalysis(name=name, providers=use_providers)
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app


def pick_largest(faces: Sequence) -> Optional:
    """Select face with largest bounding box area.

    Args:
        faces: Sequence of face detection objects with bbox attribute.

    Returns:
        Face with largest area, or None if sequence is empty.
    """
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def draw_bbox_and_label(
    frame: np.ndarray,
    bbox: Sequence[float],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw bounding box and label on frame in-place.

    Args:
        frame: BGR image to draw on.
        bbox: Bounding box as (x1, y1, x2, y2).
        label: Text label to display.
        color: BGR color tuple for box and label background.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(0, y1 - 8)
    cv2.rectangle(frame, (x1, y - th - 6), (x1 + tw + 6, y), color, -1)
    cv2.putText(
        frame, label, (x1 + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
    )


def load_projection(
    path: Path, expected_shape: Tuple[int, int] = (PROJ_DIM, EMB_DIM)
) -> np.ndarray:
    """Load projection matrix from disk with shape validation.

    Args:
        path: Path to .npy projection file.
        expected_shape: Expected matrix shape as (proj_dim, emb_dim).

    Returns:
        Projection matrix as float64 array.

    Raises:
        SystemExit: If shape mismatch detected.
    """
    P = np.load(path)
    if P.shape != expected_shape:
        raise SystemExit(
            f"[VISION] Projection shape mismatch: expected {expected_shape}, got {P.shape}"
        )
    return P.astype(np.float64, copy=False)


def get_or_make_projection(
    path: Path, proj_dim: int = PROJ_DIM, emb_dim: int = EMB_DIM
) -> np.ndarray:
    """Load or create random projection matrix.

    Args:
        path: Path to .npy projection file.
        proj_dim: Target projected dimension.
        emb_dim: Source embedding dimension.

    Returns:
        Projection matrix with shape (proj_dim, emb_dim).
    """
    if path.exists():
        return load_projection(path, (proj_dim, emb_dim))
    rng = np.random.default_rng(123)
    P = rng.normal(0, 1 / np.sqrt(proj_dim), size=(proj_dim, emb_dim)).astype(
        np.float64
    )
    np.save(path, P)
    print("[VISION] Created projection matrix at", path)
    return P


def compute_projected_embedding(P: np.ndarray, emb_512: np.ndarray) -> np.ndarray:
    """Project and normalize 512-D embedding to lower dimension.

    Args:
        P: Projection matrix with shape (proj_dim, 512).
        emb_512: Source 512-D embedding vector.

    Returns:
        Projected and L2-normalized embedding.
    """
    return l2_normalize(P @ l2_normalize(np.asarray(emb_512, dtype=np.float64)))
