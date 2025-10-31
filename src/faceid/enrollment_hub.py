"""Enrollment Hub for face identity registration.

Creates HE contexts, captures face samples, averages embeddings, and stores
enrolled identities in a database for the server to use.
"""

from __future__ import annotations

import argparse
import json
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
    get_or_make_projection,
    compute_projected_embedding,
    l2_normalize,
)
from faceid.he_utils import make_ctx, public_fingerprint
from faceid.config import (
    PUB_CTX_FILE,
    SEC_CTX_FILE,
    SERVER_DB_FILE,
    PROJ_FILE,
    EMB_DIM,
    PROJ_DIM,
)


def ensure_he_context(
    pub_path: Path = PUB_CTX_FILE, sec_path: Path = SEC_CTX_FILE
) -> bytes:
    """Ensure HE context files exist, create if missing.

    Args:
        pub_path: Path to public context file.
        sec_path: Path to secret context file.

    Returns:
        Public context bytes for fingerprint logging.
    """
    if not (pub_path.exists() and sec_path.exists()):
        _, pub_ser, sec_ser = make_ctx()
        pub_path.write_bytes(pub_ser)
        sec_path.write_bytes(sec_ser)
        print(f"[HUB] Created HE context. Public FP={public_fingerprint(pub_ser)}")
        return pub_ser

    pub_ser = pub_path.read_bytes()
    print(f"[HUB] Using existing HE context. Public FP={public_fingerprint(pub_ser)}")
    return pub_ser


def load_server_db(
    db_path: Path = SERVER_DB_FILE, expected_dim: int = PROJ_DIM
) -> Dict:
    """Load server database JSON.

    Args:
        db_path: Path to database file.
        expected_dim: Expected embedding dimension.

    Returns:
        Dict with 'names', 'embs', and 'dim' keys.
    """
    if db_path.exists():
        try:
            db = json.loads(db_path.read_text())
            if db.get("dim") != expected_dim:
                print("[HUB] DB dim mismatch; recreating DB.")
                return {"names": [], "embs": [], "dim": expected_dim}
            return db
        except Exception:
            pass
    return {"names": [], "embs": [], "dim": expected_dim}


def save_server_db(db: Dict, db_path: Path = SERVER_DB_FILE) -> None:
    """Write server database to disk.

    Args:
        db: Database dict with 'names', 'embs', 'dim'.
        db_path: Destination path.
    """
    db_path.write_text(json.dumps(db))
    print(f"[HUB] Wrote {db_path.resolve()} (copy to server)")


def compute_bbox_area(bbox: Sequence[float]) -> int:
    """Compute bounding box area.

    Args:
        bbox: Bounding box as (x1, y1, x2, y2).

    Returns:
        Area in pixels.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return max(0, x2 - x1) * max(0, y2 - y1)


def add_sample_if_valid(
    samples: List[np.ndarray], face, P: np.ndarray, min_bbox_area: int
) -> Tuple[List[np.ndarray], Optional[str]]:
    """Add face sample to buffer if bbox meets minimum size.

    Args:
        samples: Current sample buffer.
        face: InsightFace detection object.
        P: Projection matrix.
        min_bbox_area: Minimum bbox area threshold.

    Returns:
        Tuple of (updated_samples, error_message). Message is None on success.
    """
    area = compute_bbox_area(face.bbox)
    if area < min_bbox_area:
        return samples, f"[HUB] Face too small (area={area}); move closer."

    emb_red = compute_projected_embedding(P, face.normed_embedding.astype("float32"))
    return samples + [emb_red], None


def finalize_enrollment(
    db: Dict, name: str, samples: List[np.ndarray]
) -> Tuple[Dict, str]:
    """Average samples and update database.

    Args:
        db: Current database dict.
        name: Identity name to enroll.
        samples: List of embedding samples.

    Returns:
        Tuple of (updated_db, status_message).
    """
    if not samples:
        return db, "[HUB] No samples to finalize. Press 'e' to capture."

    mean = l2_normalize(np.mean(np.stack(samples, axis=0), axis=0))

    if name in db["names"]:
        idx = db["names"].index(name)
        db["embs"][idx] = mean.tolist()
        msg = f"[HUB] Finalized (avg {len(samples)}) and updated {name}."
    else:
        db["names"].append(name)
        db["embs"].append(mean.tolist())
        msg = f"[HUB] Finalized (avg {len(samples)}) and enrolled {name}."

    db["dim"] = PROJ_DIM
    return db, msg


def build_hud(
    frame: np.ndarray,
    bbox: Optional[Sequence[float]],
    label: str,
    fps: float,
    src_txt: str,
) -> np.ndarray:
    """Create HUD overlay on frame.

    Args:
        frame: Source BGR frame.
        bbox: Optional face bounding box.
        label: Text label for face.
        fps: Current FPS.
        src_txt: Source description string.

    Returns:
        New frame with HUD overlay.
    """
    disp = frame.copy()
    if bbox is not None:
        draw_bbox_and_label(disp, bbox, label)

    cv2.putText(
        disp,
        f"FPS:{fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        disp,
        f"Source: {src_txt}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 255),
        2,
    )
    return disp


def run_hub(
    enroll_name: str,
    input_mode: str,
    udp_url: str,
    cam_index: int,
    width: Optional[int],
    height: Optional[int],
    avg_samples: int,
    min_bbox_area: int,
    device: str,
) -> None:
    """Run enrollment hub main loop.

    Args:
        enroll_name: Name to enroll or update.
        input_mode: "udp" or "camera".
        udp_url: UDP stream URL.
        cam_index: Camera device index.
        width: Optional frame width.
        height: Optional frame height.
        avg_samples: Number of samples to average.
        min_bbox_area: Minimum bbox area threshold.
        device: Device mode for face detection.
    """
    ensure_he_context()
    P = get_or_make_projection(PROJ_FILE)
    db = load_server_db()
    app = ensure_face_app(device=device)

    samples: List[np.ndarray] = []
    print(
        "[HUB] Press 'e' to capture, 'f' to finalize early, 'r' to reset, 'q' to quit."
    )

    cap = open_capture(input_mode, udp_url, cam_index)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not warmup_capture(cap):
        print("[HUB] Camera/UDP read failed. Check FFmpeg/OpenCV setup.")
        return

    t_prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            if input_mode == "udp":
                cap = reconnect_capture_if_needed(
                    cap, input_mode, udp_url, cam_index, width, height
                )
                continue
            print("[HUB] Camera read failed.")
            break

        faces = app.get(frame)
        f = pick_largest(faces)
        label = f"Ready to enroll: {enroll_name}" if f is not None else "No face"
        bbox = f.bbox if f is not None else None

        t_now = time.time()
        fps = 1.0 / (t_now - t_prev) if t_now > t_prev else 0.0
        t_prev = t_now

        src_txt = udp_url if input_mode == "udp" else f"cam:{cam_index}"
        disp = build_hud(frame, bbox, label, fps, src_txt)
        cv2.imshow("Enrollment Hub", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("s"):
            fn = f"hub_snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fn, disp)
            print("[HUB] Saved", fn)
        elif k == ord("r"):
            samples = []
            print(f"[HUB] Reset samples for {enroll_name}.")
        elif k == ord("f"):
            db, msg = finalize_enrollment(db, enroll_name, samples)
            print(msg)
            if "Finalized" in msg:
                save_server_db(db)
                samples = []
        elif k == ord("e") and f is not None:
            samples, msg = add_sample_if_valid(samples, f, P, min_bbox_area)
            if msg:
                print(msg)
            else:
                print(
                    f"[HUB] Captured sample {len(samples)}/{avg_samples} for {enroll_name}."
                )
                if len(samples) >= avg_samples:
                    db, msg = finalize_enrollment(db, enroll_name, samples)
                    print(msg)
                    save_server_db(db)
                    samples = []

    cap.release()
    cv2.destroyAllWindows()


def parse_args(argv: Optional[Sequence[str]] = None) -> Dict:
    """Parse CLI arguments.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Dict of parsed arguments.
    """
    ap = argparse.ArgumentParser(description="Enrollment Hub")
    ap.add_argument(
        "--enroll-name", type=str, required=True, help="Name to enroll/update"
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
    ap.add_argument("--avg-samples", type=int, default=5, help="Samples to average")
    ap.add_argument("--min-bbox-area", type=int, default=9000, help="Min bbox area")
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Inference device",
    )

    a = ap.parse_args(argv)
    return {
        "enroll_name": a.enroll_name,
        "input_mode": a.input,
        "udp_url": a.udp_url,
        "cam_index": a.source,
        "width": a.width,
        "height": a.height,
        "avg_samples": a.avg_samples,
        "min_bbox_area": a.min_bbox_area,
        "device": a.device,
    }


def main() -> None:
    """Entry point for enrollment hub."""
    run_hub(**parse_args())


if __name__ == "__main__":
    main()
