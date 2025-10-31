"""Camera I/O utilities for video capture with UDP and local camera support.

This module provides pure functional wrappers for OpenCV VideoCapture operations,
including connection management and frame reading with automatic reconnection.

Supports two modes:
- "camera": Direct local camera access (normal use)
- "udp": UDP stream from FFmpeg (WSL/network approach)
"""

from __future__ import annotations

from typing import Optional, Tuple
import time
import cv2

__all__ = [
    "open_capture",
    "configure_capture",
    "warmup_capture",
    "reconnect_capture_if_needed",
    "read_frame_or_reconnect",
]


def open_capture(input_mode: str, udp_url: str, camera_index: int) -> cv2.VideoCapture:
    """Open a video capture from UDP stream or local camera.

    Args:
        input_mode: Either "udp" (WSL/network) or "camera" (normal local).
        udp_url: UDP URL for streaming (e.g., "udp://@0.0.0.0:5000").
        camera_index: Camera device index for local capture (typically 0).

    Returns:
        Configured VideoCapture with minimal buffer size.
    """
    if input_mode == "udp":
        print(f"[IO] Opening UDP source (WSL mode): {udp_url}")
        cap = cv2.VideoCapture(udp_url, cv2.CAP_FFMPEG)
    else:
        print(f"[IO] Opening local camera (normal mode): index {camera_index}")
        cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap


def configure_capture(
    cap: cv2.VideoCapture, width: Optional[int], height: Optional[int]
) -> cv2.VideoCapture:
    """Apply optional resolution hints to capture.

    Args:
        cap: VideoCapture to configure.
        width: Optional frame width hint.
        height: Optional frame height hint.

    Returns:
        The same capture object for chaining.
    """
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def warmup_capture(
    cap: cv2.VideoCapture, tries: int = 30, sleep_s: float = 0.1
) -> bool:
    """Attempt to read frames until source is live.

    Args:
        cap: VideoCapture to warm up.
        tries: Maximum number of read attempts.
        sleep_s: Delay between attempts in seconds.

    Returns:
        True if a frame was successfully read, False otherwise.
    """
    for _ in range(tries):
        ok, _ = cap.read()
        if ok:
            return True
        time.sleep(sleep_s)
    return False


def reconnect_capture_if_needed(
    cap: cv2.VideoCapture,
    input_mode: str,
    udp_url: str,
    camera_index: int,
    width: Optional[int],
    height: Optional[int],
) -> cv2.VideoCapture:
    """Reconnect capture after failure.

    Args:
        cap: Current capture to release.
        input_mode: "udp" or "camera".
        udp_url: UDP URL for streaming.
        camera_index: Camera device index.
        width: Optional frame width hint.
        height: Optional frame height hint.

    Returns:
        New configured VideoCapture.
    """
    try:
        cap.release()
    except Exception:
        pass
    time.sleep(0.2)
    return configure_capture(
        open_capture(input_mode, udp_url, camera_index), width, height
    )


def read_frame_or_reconnect(
    cap: cv2.VideoCapture,
    input_mode: str,
    udp_url: str,
    camera_index: int,
    width: Optional[int],
    height: Optional[int],
) -> Tuple[bool, any, cv2.VideoCapture]:
    """Read frame with automatic reconnection on failure.

    Args:
        cap: Current VideoCapture.
        input_mode: "udp" or "camera".
        udp_url: UDP URL for streaming.
        camera_index: Camera device index.
        width: Optional frame width hint.
        height: Optional frame height hint.

    Returns:
        Tuple of (success, frame, capture) where capture may be new if reconnected.
    """
    ok, frame = cap.read()
    if ok:
        return True, frame, cap
    cap = reconnect_capture_if_needed(
        cap, input_mode, udp_url, camera_index, width, height
    )
    ok, frame = cap.read()
    return ok, frame, cap
