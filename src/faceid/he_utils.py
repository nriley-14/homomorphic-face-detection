"""Homomorphic encryption utilities for CKKS-based face recognition.

This module provides pure functional wrappers around TenSEAL CKKS operations
for creating contexts, encrypting probes, and decrypting scores.
"""

from typing import Iterable, List, Tuple
import hashlib
import numpy as np
import tenseal as ts

__all__ = [
    "public_fingerprint",
    "make_ctx",
    "context_from_secret",
    "encrypt_probe",
    "decrypt_scores",
]


def public_fingerprint(pub_bytes: bytes) -> str:
    """Compute a short fingerprint for a public TenSEAL context.

    Args:
        pub_bytes: Serialized public context bytes.

    Returns:
        First 16 hex characters of SHA-256 hash.
    """
    return hashlib.sha256(pub_bytes).hexdigest()[:16]


def make_ctx(
    poly_mod_degree: int = 4096,
    coeff_mod_bit_sizes: Tuple[int, int, int] = (40, 20, 40),
    global_scale: float = 2**20,
) -> Tuple[ts.Context, bytes, bytes]:
    """Create a CKKS TenSEAL context with Galois and relin keys.

    Args:
        poly_mod_degree: Polynomial modulus degree for CKKS.
        coeff_mod_bit_sizes: Bit sizes for coefficient modulus chain.
        global_scale: Default scale for CKKS encoding.

    Returns:
        Tuple of (context, public_serialized, secret_serialized).
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, list(coeff_mod_bit_sizes)
    )
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    ctx.global_scale = global_scale
    pub = ctx.serialize(
        save_public_key=True,
        save_secret_key=False,
        save_galois_keys=True,
        save_relin_keys=True,
    )
    sec = ctx.serialize(
        save_public_key=True,
        save_secret_key=True,
        save_galois_keys=True,
        save_relin_keys=True,
    )
    return ctx, pub, sec


def context_from_secret(secret_bytes: bytes) -> ts.Context:
    """Deserialize a TenSEAL context from secret bytes.

    Args:
        secret_bytes: Serialized context containing secret key.

    Returns:
        Reconstructed TenSEAL context.
    """
    return ts.context_from(secret_bytes)


def encrypt_probe(ctx: ts.Context, probe_128: np.ndarray) -> bytes:
    """Encrypt a 128-D probe vector using CKKS.

    Args:
        ctx: TenSEAL context with public key.
        probe_128: Normalized 128-D embedding vector.

    Returns:
        Serialized encrypted CKKS vector.
    """
    enc_vec = ts.ckks_vector(ctx, np.asarray(probe_128, dtype=float).tolist())
    return enc_vec.serialize()


def decrypt_scores(ctx: ts.Context, scores_enc: Iterable[bytes]) -> List[float]:
    """Decrypt and clip encrypted similarity scores.

    Args:
        ctx: TenSEAL context with secret key.
        scores_enc: Iterable of serialized encrypted score vectors.

    Returns:
        List of decrypted scores clipped to [-1.0, 1.0].
    """
    return [
        max(-1.0, min(1.0, float(ts.ckks_vector_from(ctx, blob).decrypt()[0])))
        for blob in scores_enc
    ]
