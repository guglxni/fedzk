# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""Circuit capacity configuration for FEDzk v2 (Phase 0/1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# Shipped wasm / prove path capacity (Bolt 1).
N_DEV = 4

# Phase 1 targets (artifacts not yet regenerated).
N_TARGETS: Tuple[int, ...] = (4, 64, 256)

DEFAULT_SCALE_FACTOR = 1000
CIRCUIT_ID_STANDARD = "model_update"
CIRCUIT_ID_SECURE = "model_update_secure"
WIRE_FORMAT = "fedzk.proof.v1"


@dataclass(frozen=True)
class CircuitProfile:
    circuit_id: str
    n: int
    secure: bool
    scale_factor: int = DEFAULT_SCALE_FACTOR

    @property
    def wasm_name(self) -> str:
        return "model_update_secure.wasm" if self.secure else "model_update.wasm"

    @property
    def zkey_name(self) -> str:
        return "proving_key_secure.zkey" if self.secure else "proving_key.zkey"

    @property
    def vkey_name(self) -> str:
        return "verification_key_secure.json" if self.secure else "verification_key.json"


PROFILE_N4 = CircuitProfile(CIRCUIT_ID_STANDARD, N_DEV, secure=False)
PROFILE_N4_SECURE = CircuitProfile(CIRCUIT_ID_SECURE, N_DEV, secure=True)


def require_supported_n(n: int) -> None:
    """Fail closed if requesting an N without shipped artifacts."""
    if n != N_DEV:
        raise ValueError(
            f"Circuit capacity N={n} not shipped yet (N_dev={N_DEV}). "
            f"Phase 1 targets {list(N_TARGETS)}; regenerate artifacts first."
        )
