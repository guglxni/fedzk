# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""Circuit capacity configuration for FEDzk v2 (Phase 0/1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

# Legacy frozen prove path (Bolt 1) — snarkjs-shaped gradients[4].
N_DEV = 4

# Phase 1: ModelUpdateGradients(N) with shipped (dev-ceremony) artifacts for 64.
N_TARGETS: Tuple[int, ...] = (4, 64, 256)

DEFAULT_SCALE_FACTOR = 1000
CIRCUIT_ID_STANDARD = "model_update"
CIRCUIT_ID_SECURE = "model_update_secure"
CIRCUIT_ID_GRADIENTS_N64 = "model_update_gradients_n64"
CIRCUIT_ID_GRADIENTS_N256 = "model_update_gradients_n256"
WIRE_FORMAT = "fedzk.proof.v1"

# N values that have *runtime* artifacts under src/fedzk/zk/
SHIPPED_N: Tuple[int, ...] = (4, 64, 256)


@dataclass(frozen=True)
class CircuitProfile:
    circuit_id: str
    n: int
    secure: bool
    scale_factor: int = DEFAULT_SCALE_FACTOR
    wasm_name: str = "model_update.wasm"
    zkey_name: str = "proving_key.zkey"
    vkey_name: str = "verification_key.json"
    ceremony: str = "frozen"  # frozen | dev_unsafe | production

    def resolve(self, asset_dir: Path) -> Dict[str, Path]:
        return {
            "wasm": asset_dir / self.wasm_name,
            "zkey": asset_dir / self.zkey_name,
            "vkey": asset_dir / self.vkey_name,
        }


PROFILE_N4 = CircuitProfile(
    CIRCUIT_ID_STANDARD,
    N_DEV,
    secure=False,
    ceremony="frozen",
)
PROFILE_N4_SECURE = CircuitProfile(
    CIRCUIT_ID_SECURE,
    N_DEV,
    secure=True,
    wasm_name="model_update_secure.wasm",
    zkey_name="proving_key_secure.zkey",
    vkey_name="verification_key_secure.json",
    ceremony="frozen",
)
PROFILE_N64 = CircuitProfile(
    CIRCUIT_ID_GRADIENTS_N64,
    64,
    secure=False,
    wasm_name="model_update_gradients_n64.wasm",
    zkey_name="proving_key_gradients_n64.zkey",
    vkey_name="verification_key_gradients_n64.json",
    ceremony="dev_unsafe",
)
PROFILE_N256 = CircuitProfile(
    CIRCUIT_ID_GRADIENTS_N256,
    256,
    secure=False,
    wasm_name="model_update_gradients_n256.wasm",
    zkey_name="proving_key_gradients_n256.zkey",
    vkey_name="verification_key_gradients_n256.json",
    ceremony="dev_unsafe",
)

PROFILES_BY_ID = {
    PROFILE_N4.circuit_id: PROFILE_N4,
    PROFILE_N4_SECURE.circuit_id: PROFILE_N4_SECURE,
    PROFILE_N64.circuit_id: PROFILE_N64,
    PROFILE_N256.circuit_id: PROFILE_N256,
}


def profile_for_n(n: int, secure: bool = False) -> CircuitProfile:
    if n == 4 and secure:
        return PROFILE_N4_SECURE
    if n == 4:
        return PROFILE_N4
    if n == 64 and not secure:
        return PROFILE_N64
    if n == 256 and not secure:
        return PROFILE_N256
    raise ValueError(
        f"No CircuitProfile for n={n} secure={secure}. Shipped: {list(SHIPPED_N)}."
    )


def require_supported_n(n: int) -> None:
    """Fail closed if requesting an N without shipped artifacts."""
    if n not in SHIPPED_N:
        raise ValueError(
            f"Circuit capacity N={n} not shipped yet (shipped={list(SHIPPED_N)}). "
            f"Phase 1 targets {list(N_TARGETS)}; regenerate artifacts first."
        )
