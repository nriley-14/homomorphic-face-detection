"""HE Server for privacy-preserving face recognition scoring.

This server receives encrypted face embeddings, computes homomorphic dot products
against a database of enrolled identities, and returns encrypted similarity scores.
All operations preserve privacy by working on encrypted data.
"""

import json
import socket
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tenseal as ts

from faceid.he_utils import public_fingerprint
from faceid.config import (
    SERVER_DB_FILE as DB_FILE,
    PROJ_FILE,
    PUB_CTX_FILE as PUB_FILE,
    SEC_CTX_FILE as SECRET_FILE,
    THRESHOLD,
    SERVER_HOST as HOST,
    SERVER_PORT as PORT,
)


def public_from_secret_ctx(ctx_secret: ts.Context) -> bytes:
    """Serialize context as public-only blob for fingerprint comparison.

    Args:
        ctx_secret: TenSEAL context with secret key.

    Returns:
        Serialized public context without secret key.
    """
    return ctx_secret.serialize(
        save_public_key=True,
        save_secret_key=False,
        save_galois_keys=True,
        save_relin_keys=True,
    )


def load_db() -> Dict:
    """Load face embedding database with validation.

    Returns:
        Dict with keys 'names', 'embs', and 'dim'. Returns empty structure if missing.
    """
    if not DB_FILE.exists():
        print("[SERVER] Missing DB. Copy faceid_db_server.json from hub.")
        return {"names": [], "embs": [], "dim": 128}

    db = json.loads(DB_FILE.read_text())

    if PROJ_FILE.exists():
        try:
            P = np.load(PROJ_FILE)
            if P.shape[0] != db.get("dim", 128):
                print("[SERVER] WARNING: he_proj.npy dim mismatch vs DB.")
        except Exception as e:
            print("[SERVER] WARNING: failed to load he_proj.npy:", e)
    else:
        print("[SERVER] Missing he_proj.npy. Copy from hub (optional).")

    return db


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from socket.

    Args:
        sock: Connected socket.
        n: Number of bytes to receive.

    Returns:
        Received bytes of length n.

    Raises:
        ConnectionError: If socket closes before n bytes received.
    """
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


def recv_u32(sock: socket.socket) -> int:
    """Receive big-endian unsigned 32-bit integer.

    Args:
        sock: Connected socket.

    Returns:
        Decoded integer value.
    """
    return struct.unpack("!I", recv_exact(sock, 4))[0]


def recv_blob(sock: socket.socket) -> bytes:
    """Receive length-prefixed byte blob.

    Args:
        sock: Connected socket.

    Returns:
        Received byte blob.
    """
    length = recv_u32(sock)
    return recv_exact(sock, length) if length > 0 else b""


def send_u32(sock: socket.socket, n: int) -> None:
    """Send big-endian unsigned 32-bit integer.

    Args:
        sock: Connected socket.
        n: Integer value to send.
    """
    sock.sendall(struct.pack("!I", n))


def send_blob(sock: socket.socket, b: bytes) -> None:
    """Send length-prefixed byte blob.

    Args:
        sock: Connected socket.
        b: Bytes to send.
    """
    send_u32(sock, len(b))
    if b:
        sock.sendall(b)


def load_contexts() -> Tuple[Optional[ts.Context], Optional[ts.Context]]:
    """Load cached public context from disk. Server operates in blind mode.

    Returns:
        Tuple of (public_context, None). Secret context never loaded on server.
    """
    ctx_pub = None

    if PUB_FILE.exists():
        try:
            ctx_pub = ts.context_from(PUB_FILE.read_bytes())
            print("[SERVER] Loaded cached public context.")
        except Exception as e:
            print("[SERVER] Failed to load public ctx:", e)

    return ctx_pub, None


def validate_client_context(pubblob: bytes, ctx_secret: Optional[ts.Context]) -> bool:
    """Validate client public context against server secret context.

    Args:
        pubblob: Serialized public context from client.
        ctx_secret: Server's secret context for comparison.

    Returns:
        True if contexts match or no secret available, False on mismatch.
    """
    if ctx_secret is None:
        return True

    cam_fp = public_fingerprint(pubblob)
    srv_fp = public_fingerprint(public_from_secret_ctx(ctx_secret))

    if cam_fp != srv_fp:
        print(
            f"[SERVER] ERROR: HE context mismatch! camera FP={cam_fp} server FP={srv_fp}"
        )
        return False

    return True


def update_public_ctx_from_client(
    conn: socket.socket, ctx_secret: Optional[ts.Context]
) -> Optional[ts.Context]:
    """Receive and validate public context from client.

    Args:
        conn: Connected client socket.
        ctx_secret: Server's secret context for validation.

    Returns:
        New public context if provided and valid, None otherwise.
    """
    publen = recv_u32(conn)
    if publen == 0:
        return None

    pubblob = recv_exact(conn, publen)
    PUB_FILE.write_bytes(pubblob)
    print(
        f"[SERVER] Public context updated from camera. FP={public_fingerprint(pubblob)}"
    )

    if not validate_client_context(pubblob, ctx_secret):
        return None

    try:
        return ts.context_from(pubblob)
    except Exception as e:
        print("[SERVER] Failed to parse camera public context:", e)
        return None


def receive_detection_result(conn: socket.socket) -> None:
    """Receive detection result from camera and log it.

    Args:
        conn: Connected client socket.
    """
    try:
        raw = conn.recv(4)
        if len(raw) < 4:
            return
        (length,) = struct.unpack("!I", raw)
        
        if length == 0 or length > 1024:  # Sanity check
            return
        
        result_bytes = recv_exact(conn, length)
        result = json.loads(result_bytes.decode('utf-8'))
        
        name = result.get("name", "Unknown")
        present = result.get("present", False)
        
        if present:
            print(f"[SERVER] Detection logged: {name} -> PRESENT")
        else:
            print(f"[SERVER] Detection logged: {name} -> NOT PRESENT")
    except Exception:
        pass  # Client may have disconnected


def compute_encrypted_scores(
    ctx_pub: ts.Context, probe_blob: bytes, db_embs: List[List[float]]
) -> List[bytes]:
    """Compute encrypted dot products between probe and database embeddings.

    Args:
        ctx_pub: Public TenSEAL context.
        probe_blob: Serialized encrypted probe vector.
        db_embs: Plain embeddings from database.

    Returns:
        List of serialized encrypted score vectors.
    """
    enc_probe = ts.ckks_vector_from(ctx_pub, probe_blob)
    return [enc_probe.dot(emb).serialize() for emb in db_embs]


def decrypt_and_log_best(
    ctx_secret: Optional[ts.Context],
    scores_enc: List[bytes],
    names: List[str],
) -> None:
    """Decrypt scores and log best match if secret context available.

    Args:
        ctx_secret: Secret context for decryption, None disables logging.
        scores_enc: Serialized encrypted scores.
        names: Parallel list of identity names.
    """
    if ctx_secret is None or not scores_enc:
        return

    scores = [
        float(ts.ckks_vector_from(ctx_secret, es).decrypt()[0]) for es in scores_enc
    ]
    if not scores:
        return

    j = int(np.argmax(scores))
    best_name, best_score = names[j], scores[j]

    if abs(best_score) > 2.0:
        print(
            f"[SERVER] WARNING: suspicious score {best_score:.3f}. Check normalization."
        )
    elif best_score >= THRESHOLD:
        print(
            f"[SERVER] Person detected: {best_name} (score {best_score:.3f}) -> PRESENT"
        )


def handle_client(
    conn: socket.socket,
    cached_ctx_pub: Optional[ts.Context],
    ctx_secret: Optional[ts.Context],
    db: Dict,
) -> Optional[ts.Context]:
    """Handle single client request and return updated public context.

    Args:
        conn: Connected client socket.
        cached_ctx_pub: Previously cached public context.
        ctx_secret: Secret context for logging (unused - server is blind).
        db: Loaded database dict.

    Returns:
        Public context to cache for subsequent clients.
    """
    new_ctx_pub = update_public_ctx_from_client(conn, ctx_secret)
    if new_ctx_pub is None and cached_ctx_pub is None:
        send_u32(conn, 0)
        return cached_ctx_pub

    ctx_pub = new_ctx_pub or cached_ctx_pub

    if new_ctx_pub is None and PUB_FILE.exists() and cached_ctx_pub is None:
        send_u32(conn, 0)
        return cached_ctx_pub

    probe_len = recv_u32(conn)
    probe_blob = recv_exact(conn, probe_len) if probe_len > 0 else b""

    scores_enc = compute_encrypted_scores(ctx_pub, probe_blob, db["embs"])

    send_u32(conn, len(scores_enc))
    for es in scores_enc:
        send_blob(conn, es)

    # Receive detection result from camera for logging
    receive_detection_result(conn)

    return ctx_pub


def main() -> None:
    """Run the FaceID HE scoring server.

    Loads database and contexts, listens for client connections, and processes
    encrypted face recognition requests using homomorphic encryption.
    """
    ctx_pub, ctx_secret = load_contexts()
    db = load_db()

    if not db.get("embs"):
        print("[SERVER] WARNING: DB empty. Populate from hub.")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(8)
        print(f"[SERVER] Listening on {HOST}:{PORT}")

        while True:
            conn, addr = srv.accept()
            with conn:
                try:
                    updated_ctx_pub = handle_client(conn, ctx_pub, ctx_secret, db)
                    if updated_ctx_pub is not None and updated_ctx_pub is not ctx_pub:
                        PUB_FILE.write_bytes(
                            updated_ctx_pub.serialize(
                                save_public_key=True,
                                save_secret_key=False,
                                save_galois_keys=True,
                                save_relin_keys=True,
                            )
                        )
                        ctx_pub = updated_ctx_pub
                except Exception as e:
                    print("[SERVER] Error handling client:", e)
                    try:
                        send_u32(conn, 0)
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
