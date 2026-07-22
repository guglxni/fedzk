# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
Chunk Protocol v1 — honest capacity beyond N_dev.

When |update| > N, split into fixed-size chunks, prove each chunk, bind with a
commitment over the full quantized vector. Coordinator must verify all chunks
+ commitment before FedAvg (Phase 1 wiring).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence

from fedzk.zk.circuit_config import N_DEV, WIRE_FORMAT


def commit_quantized(values: Sequence[int]) -> str:
    """SHA-256 commitment over the full integer vector (canonical JSON list)."""
    payload = json.dumps([int(v) for v in values], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ChunkSpec:
    circuit_id: str
    n: int
    chunk_index: int
    chunk_count: int
    commitment: str
    values: List[int]

    def to_public_meta(self) -> Dict[str, Any]:
        return {
            "circuit_id": self.circuit_id,
            "n": self.n,
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "commitment": self.commitment,
            "wire": WIRE_FORMAT,
        }


@dataclass
class ChunkProofBundle:
    """Full update as chunk proofs bound by one commitment."""

    commitment: str
    n: int
    circuit_id: str
    chunks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def split_into_chunks(
    quantized: Sequence[int],
    *,
    n: int = N_DEV,
    circuit_id: str = "model_update",
    pad_last: bool = True,
) -> List[ChunkSpec]:
    """
    Split a quantized vector into length-n chunks.

    Last chunk is zero-padded when pad_last=True (logged by caller).
    Empty input yields one all-zero chunk (explicit empty update).
    """
    values = [int(v) for v in quantized]
    commitment = commit_quantized(values)
    if not values:
        values = [0] * n
    chunk_count = max(1, (len(values) + n - 1) // n)
    specs: List[ChunkSpec] = []
    for i in range(chunk_count):
        start = i * n
        piece = values[start : start + n]
        if pad_last and len(piece) < n:
            piece = piece + [0] * (n - len(piece))
        specs.append(
            ChunkSpec(
                circuit_id=circuit_id,
                n=n,
                chunk_index=i,
                chunk_count=chunk_count,
                commitment=commitment,
                values=piece,
            )
        )
    return specs


def verify_bundle_commitments(bundle: ChunkProofBundle) -> bool:
    """Structural check: every chunk meta shares the same commitment and counts."""
    if not bundle.chunks:
        return False
    for ch in bundle.chunks:
        meta = ch.get("meta") or {}
        if meta.get("commitment") != bundle.commitment:
            return False
        if int(meta.get("chunk_count", -1)) != len(bundle.chunks):
            return False
        if int(meta.get("n", -1)) != bundle.n:
            return False
    return True
