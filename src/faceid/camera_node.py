"""Camera Node for privacy-preserving face recognition.

Captures video frames, detects faces, computes embeddings locally, encrypts them
with homomorphic encryption, and sends only ciphertexts to the server. Raw video
never leaves the device.
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from faceid.camera_io import open_capture, warmup_capture, reconnect_capture_if_needed
from faceid.vision import (
    ensure_face_app,
    pick_largest,
    draw_bbox_and_label,
    load_projection,
    compute_projected_embedding,
)
from faceid.he_utils import (
    context_from_secret,
    public_fingerprint,
    encrypt_probe,
    decrypt_scores,
)
from faceid.config import (
    PROJ_FILE,
    PUB_CTX_FILE as PUB_FILE,
    SEC_CTX_FILE as SEC_FILE,
    SERVER_DB_FILE,
    SERVER_HOST,
    SERVER_PORT,
    THRESHOLD,
)


def load_names_for_hud(db_path: Path = SERVER_DB_FILE) -> List[str]:
    """Load enrolled identity names for HUD display.

    Args:
        db_path: Path to server database JSON.

    Returns:
        List of enrolled names, empty if file missing or invalid.
    """
    if db_path.exists():
        try:
            return json.loads(db_path.read_text()).get("names", [])
        except Exception:
            return []
    return []


def load_he_state() -> Dict:
    """Load HE context and public bytes from disk.

    Returns:
        Dict with 'client_ctx' (TenSEAL context) and 'public_bytes' (bytes or None).

    Raises:
        SystemExit: If required HE files are missing.
    """
    if not (PUB_FILE.exists() and SEC_FILE.exists()):
        raise SystemExit(
            "[CAM] Missing he_public_ctx.bin / he_secret_ctx.bin. Copy from hub."
        )

    pub_ser = PUB_FILE.read_bytes()
    sec_ser = SEC_FILE.read_bytes()
    client_ctx = context_from_secret(sec_ser)
    print(f"[CAM] Loaded HE context. Public FP={public_fingerprint(pub_ser)}")
    return {"client_ctx": client_ctx, "public_bytes": pub_ser}


def send_probe_receive_scores(
    enc_probe_ser: bytes,
    public_ctx_ser: Optional[bytes],
    host: str,
    port: int,
    timeout_s: float = 2.5,
) -> Tuple[Optional[List[bytes]], Optional[socket.socket]]:
    """Send encrypted probe to server and receive encrypted scores.

    Args:
        enc_probe_ser: Serialized encrypted probe vector.
        public_ctx_ser: Optional public context to send (None to skip).
        host: Server hostname.
        port: Server port.
        timeout_s: Connection timeout in seconds.

    Returns:
        Tuple of (list of encrypted score blobs or None, socket for sending result back or None).
    """
    try:
        s = socket.create_connection((host, port), timeout=timeout_s)
        s.sendall(struct.pack("!I", len(public_ctx_ser) if public_ctx_ser else 0))
        if public_ctx_ser:
            s.sendall(public_ctx_ser)

        s.sendall(struct.pack("!I", len(enc_probe_ser)))
        s.sendall(enc_probe_ser)

        raw = s.recv(4)
        if len(raw) < 4:
            s.close()
            return None, None
        (cnt,) = struct.unpack("!I", raw)

        out: List[bytes] = []
        for _ in range(cnt):
            raw = s.recv(4)
            if len(raw) < 4:
                s.close()
                return None, None
            (n,) = struct.unpack("!I", raw)
            blob = b""
            while len(blob) < n:
                chunk = s.recv(n - len(blob))
                if not chunk:
                    s.close()
                    return None, None
                blob += chunk
            out.append(blob)
        return out, s
    except Exception:
        return None, None


def send_detection_result(
    sock: Optional[socket.socket], name: str, present: bool
) -> None:
    """Send detection result back to server for logging.

    Args:
        sock: Connected socket to server.
        name: Detected person's name.
        present: Whether threshold was met.
    """
    if sock is None:
        return

    try:
        result = json.dumps({"name": name, "present": present})
        sock.sendall(struct.pack("!I", len(result)))
        sock.sendall(result.encode("utf-8"))
    except Exception:
        pass
    finally:
        sock.close()


def update_hud_state(
    hud: Dict,
    now: float,
    scores_enc: Optional[List[bytes]],
    client_ctx,
    names: List[str],
) -> Tuple[Dict, Optional[str], bool]:
    """Update HUD state from encrypted scores.

    Args:
        hud: Current HUD state dict.
        now: Current timestamp.
        scores_enc: Encrypted scores from server, None if offline.
        client_ctx: TenSEAL context for decryption.
        names: List of enrolled names for logging.

    Returns:
        Tuple of (updated HUD state dict, detected name or None, is_present bool).
    """
    if scores_enc is None:
        return {**hud, "server_online": False}, None, False

    if not scores_enc:
        return {**hud, "server_online": True}, None, False

    scores = decrypt_scores(client_ctx, scores_enc)
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]

    # Log what server sent (encrypted) and what we decrypted
    print(f"[CAM] Received {len(scores_enc)} encrypted scores from server")
    print(f"[CAM] Decrypted scores: {[f'{s:.3f}' for s in scores]}")

    detected_name = None
    is_present = False

    if best_idx < len(names):
        best_name = names[best_idx]
        detected_name = best_name
        is_present = best_score >= THRESHOLD

        if is_present:
            print(f"[CAM] Detection: {best_name} (score {best_score:.3f}) -> PRESENT")
        else:
            print(
                f"[CAM] Best match: {best_name} (score {best_score:.3f}) -> NOT PRESENT (below threshold {THRESHOLD})"
            )

    return (
        {
            **hud,
            "server_online": True,
            "last_best_idx": best_idx,
            "last_score": best_score,
            "last_send_time": now,
        },
        detected_name,
        is_present,
    )


def should_send_probe(now: float, last_send: float, interval: float) -> bool:
    """Check if enough time has elapsed to send another probe.

    Args:
        now: Current timestamp.
        last_send: Last send timestamp.
        interval: Minimum interval between sends.

    Returns:
        True if should send, False otherwise.
    """
    return (now - last_send) >= interval


def visualize_encrypted_data(
    enc_probe_ser: bytes, size: Tuple[int, int] = (400, 400)
) -> np.ndarray:
    """Convert encrypted bytes to visual noise representation.

    Args:
        enc_probe_ser: Serialized encrypted probe bytes.
        size: Output image size (width, height).

    Returns:
        BGR image showing encrypted data as pixel noise.
    """
    total_pixels = size[0] * size[1] * 3
    byte_array = np.frombuffer(enc_probe_ser, dtype=np.uint8)

    if len(byte_array) < total_pixels:
        byte_array = np.tile(byte_array, (total_pixels // len(byte_array)) + 1)

    pixels = byte_array[:total_pixels].reshape(size[1], size[0], 3)
    return pixels.astype(np.uint8)


def build_hud_frame(
    frame: np.ndarray,
    bbox: Optional[Sequence[float]],
    label: str,
    hud: Dict,
    input_mode: str,
    camera_index: int,
    server_host: str,
    server_port: int,
    fps: float,
    device: str,
    he_send_fps: float,
) -> np.ndarray:
    """Compose HUD overlay on frame.

    Args:
        frame: Source BGR frame.
        bbox: Optional face bounding box.
        label: Text label for face.
        hud: HUD state dict.
        input_mode: "udp" or "camera".
        camera_index: Camera device index.
        server_host: Server hostname.
        server_port: Server port.
        fps: Current FPS.
        device: Device mode string.
        he_send_fps: Probe send frequency.

    Returns:
        New frame with HUD overlay.
    """
    disp = frame.copy()
    if bbox is not None:
        color = (0, 255, 0) if hud["server_online"] else (0, 165, 255)
        draw_bbox_and_label(disp, bbox, label, color)

    status = "udp" if input_mode == "udp" else f"cam:{camera_index}"
    server_status = "ONLINE" if hud["server_online"] else "OFFLINE"
    server_color = (0, 255, 0) if hud["server_online"] else (0, 165, 255)

    cv2.putText(
        disp,
        f"FPS:{fps:.1f}  send@{he_send_fps:.2f}Hz  dev:{device}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        disp,
        f"Source: {status}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 255),
        2,
    )
    cv2.putText(
        disp,
        f"Server: {server_status} @ {server_host}:{server_port}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        server_color,
        2,
    )
    return disp


def compute_label_from_hud(hud: Dict, names: List[str]) -> str:
    """Generate face label from HUD state.

    Args:
        hud: HUD state dict.
        names: List of enrolled names.

    Returns:
        Label string for display (best match score, for demo only).
    """
    if hud.get("last_best_idx") is None or hud.get("last_score") is None:
        return "Detecting…"

    idx = hud["last_best_idx"]
    best_label = names[idx] if idx < len(names) else f"#{idx}"
    return f"{best_label}: {hud['last_score']:.2f}"


def run_camera_node(
    input_mode: str,
    udp_url: str,
    camera_index: int,
    width: Optional[int],
    height: Optional[int],
    server_host: str,
    server_port: int,
    he_send_fps: float,
    device: str,
) -> None:
    """Run camera node main loop.

    Args:
        input_mode: "udp" or "camera".
        udp_url: UDP stream URL.
        camera_index: Camera device index.
        width: Optional frame width.
        height: Optional frame height.
        server_host: Server hostname.
        server_port: Server port.
        he_send_fps: Probe send frequency in Hz.
        device: Device mode for face detection.
    """
    print("[CAM] No frames leave device. Only encrypted embeddings sent.")

    P = load_projection(PROJ_FILE)
    names = load_names_for_hud()
    app = ensure_face_app(device=device)
    he_state = load_he_state()

    cap = open_capture(input_mode, udp_url, camera_index)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not warmup_capture(cap):
        print("[CAM] Initial read failed. Check FFmpeg/OpenCV setup.")
        return

    hud = {
        "server_online": False,
        "last_best_idx": None,
        "last_score": None,
        "last_send_time": 0.0,
    }
    send_interval = 1.0 / max(1e-6, he_send_fps)
    i, t_prev, last_face = 0, time.time(), None
    show_encrypted = False
    last_encrypted_bytes = None

    while True:
        ok, frame = cap.read()
        if not ok:
            if input_mode == "udp":
                cap = reconnect_capture_if_needed(
                    cap, input_mode, udp_url, camera_index, width, height
                )
                continue
            print("[CAM] Camera read failed.")
            break

        if i % 3 == 0:
            faces = app.get(frame)
            last_face = pick_largest(faces)

        label, bbox = "No face", None

        if last_face is not None:
            bbox = last_face.bbox
            label = "Face detected"

            # Show name if server confirmed PRESENT (score >= threshold)
            if hud.get("last_score") is not None and hud["last_score"] >= THRESHOLD:
                idx = hud.get("last_best_idx")
                if idx is not None and idx < len(names):
                    label = names[idx]

            now = time.time()

            if should_send_probe(now, hud["last_send_time"], send_interval):
                probe = compute_projected_embedding(
                    P, last_face.normed_embedding.astype("float32")
                )
                enc_probe = encrypt_probe(he_state["client_ctx"], probe)
                last_encrypted_bytes = enc_probe
                scores_enc, conn_sock = send_probe_receive_scores(
                    enc_probe, he_state["public_bytes"], server_host, server_port
                )
                he_state["public_bytes"] = None
                hud, detected_name, is_present = update_hud_state(
                    hud, now, scores_enc, he_state["client_ctx"], names
                )

                # Send detection result back to server for logging
                if detected_name is not None:
                    send_detection_result(conn_sock, detected_name, is_present)
                elif conn_sock is not None:
                    conn_sock.close()

        t_now = time.time()
        fps = 1.0 / (t_now - t_prev) if t_now > t_prev else 0.0
        t_prev = t_now

        disp = build_hud_frame(
            frame,
            bbox,
            label,
            hud,
            input_mode,
            camera_index,
            server_host,
            server_port,
            fps,
            device,
            he_send_fps,
        )
        cv2.imshow("Camera Node (Encrypted Probes Only)", disp)

        if show_encrypted and last_encrypted_bytes is not None:
            enc_vis = visualize_encrypted_data(last_encrypted_bytes)
            cv2.imshow("Encrypted Data (What Leaves the Camera)", enc_vis)
        elif not show_encrypted:
            cv2.destroyWindow("Encrypted Data (What Leaves the Camera)")

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("s"):
            fn = f"cam_snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fn, disp)
            print("[CAM] Saved", fn)
        elif k == ord("v"):
            show_encrypted = not show_encrypted
            print(f"[CAM] Encrypted visualization: {'ON' if show_encrypted else 'OFF'}")

        i += 1

    cap.release()
    cv2.destroyAllWindows()


def parse_args(argv: Optional[Sequence[str]] = None) -> Dict:
    """Parse CLI arguments.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Dict of parsed arguments.
    """
    ap = argparse.ArgumentParser(description="Camera node: encrypted probe sender")
    ap.add_argument(
        "--he-send-fps", type=float, default=1.0, help="Probe send frequency (Hz)"
    )
    ap.add_argument(
        "--input",
        type=str,
        default="udp",
        choices=["udp", "camera"],
        help="Input source",
    )
    ap.add_argument(
        "--udp-url", type=str, default="udp://@0.0.0.0:5000", help="UDP URL"
    )
    ap.add_argument("--source", type=int, default=0, help="Camera index")
    ap.add_argument("--width", type=int, default=None, help="Frame width")
    ap.add_argument("--height", type=int, default=None, help="Frame height")
    ap.add_argument("--server-host", type=str, default=SERVER_HOST, help="Server host")
    ap.add_argument("--server-port", type=int, default=SERVER_PORT, help="Server port")
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Inference device",
    )

    a = ap.parse_args(argv)
    return {
        "input_mode": a.input,
        "udp_url": a.udp_url,
        "camera_index": a.source,
        "width": a.width,
        "height": a.height,
        "server_host": a.server_host,
        "server_port": a.server_port,
        "he_send_fps": a.he_send_fps,
        "device": a.device,
    }


def main() -> None:
    """Entry point for camera node."""
    run_camera_node(**parse_args())


if __name__ == "__main__":
    main()
