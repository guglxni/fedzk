# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""Chunked prove helper — honest capacity beyond single-circuit N."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import torch

from fedzk.prover.zkgenerator import ASSET_DIR, ZKProver
from fedzk.zk.chunk_protocol import ChunkProofBundle, commit_quantized, split_into_chunks
from fedzk.zk.circuit_config import N_DEV, PROFILE_N4, CircuitProfile, profile_for_n
from fedzk.zk.input_normalization import GradientQuantizer


def _flatten_quantized(gradient_dict: Dict[str, torch.Tensor], scale_factor: int) -> List[int]:
    q = GradientQuantizer(scale_factor=scale_factor)
    quantized, _ = q.quantize_gradients(gradient_dict)
    flat: List[int] = []
    for values in quantized.values():
        flat.extend(int(v) for v in values)
    return flat


def _prove_integer_vector(
    values: Sequence[int],
    profile: CircuitProfile,
    asset_dir: Path = ASSET_DIR,
) -> Dict[str, Any]:
    """Prove a length-profile.n integer vector with the profile's wasm/zkey."""
    if len(values) != profile.n:
        raise ValueError(f"Expected {profile.n} values, got {len(values)}")
    paths = profile.resolve(asset_dir)
    for k, p in paths.items():
        if k == "vkey":
            continue
        if not p.is_file():
            raise FileNotFoundError(f"Missing {k} artifact: {p}")

    input_data = {"gradients": [int(v) for v in values]}
    with tempfile.TemporaryDirectory() as tmpdir:
        input_json = os.path.join(tmpdir, "input.json")
        witness = os.path.join(tmpdir, "witness.wtns")
        proof_path = os.path.join(tmpdir, "proof.json")
        public_path = os.path.join(tmpdir, "public.json")
        with open(input_json, "w") as f:
            json.dump(input_data, f)
        subprocess.run(
            ["snarkjs", "wtns", "calculate", str(paths["wasm"]), input_json, witness],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "snarkjs",
                "groth16",
                "prove",
                str(paths["zkey"]),
                witness,
                proof_path,
                public_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with open(proof_path) as f:
            proof = json.load(f)
        with open(public_path) as f:
            public_inputs = json.load(f)
    return {
        "proof": proof,
        "public_inputs": public_inputs,
        "circuit_id": profile.circuit_id,
        "n": profile.n,
        "ceremony": profile.ceremony,
        "wire": "fedzk.proof.v1",
    }


def prove_update(
    gradient_dict: Dict[str, torch.Tensor],
    *,
    n: int = N_DEV,
    scale_factor: int = 1000,
    prefer_single_if_fits: bool = True,
) -> Dict[str, Any]:
    """
    Prove a model update.

    - If len(quantized) <= n and prefer_single_if_fits: one proof.
    - Else: Chunk Protocol v1 bundle (multiple proofs + commitment).
    """
    flat = _flatten_quantized(gradient_dict, scale_factor)
    profile = profile_for_n(n, secure=False)
    commitment = commit_quantized(flat)

    if prefer_single_if_fits and len(flat) <= n:
        padded = flat + [0] * (n - len(flat))
        single = _prove_integer_vector(padded, profile)
        single["commitment"] = commitment
        single["mode"] = "single"
        single["quantized_len"] = len(flat)
        return single

    specs = split_into_chunks(flat, n=n, circuit_id=profile.circuit_id)
    chunks: List[Dict[str, Any]] = []
    for spec in specs:
        part = _prove_integer_vector(spec.values, profile)
        chunks.append(
            {
                "meta": spec.to_public_meta(),
                "proof": part["proof"],
                "public_inputs": part["public_inputs"],
            }
        )
    bundle = ChunkProofBundle(
        commitment=commitment,
        n=n,
        circuit_id=profile.circuit_id,
        chunks=chunks,
    )
    out = bundle.to_dict()
    out["mode"] = "chunked"
    out["quantized_len"] = len(flat)
    out["wire"] = "fedzk.proof.v1"
    out["ceremony"] = profile.ceremony
    return out


def prove_with_legacy_prover(gradient_dict: Dict[str, torch.Tensor], secure: bool = False) -> Dict[str, Any]:
    """Backward-compatible: ZKProver N_dev=4 path."""
    prover = ZKProver(secure=secure)
    proof, public_inputs = prover.generate_proof(gradient_dict)
    return {
        "proof": proof,
        "public_inputs": public_inputs,
        "mode": "legacy_n4",
        "circuit_id": PROFILE_N4.circuit_id if not secure else "model_update_secure",
        "n": N_DEV,
        "wire": "fedzk.proof.v1",
        "quantization": getattr(prover, "last_quantization_metadata", {}),
    }
