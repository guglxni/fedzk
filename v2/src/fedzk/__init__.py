# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""FEDzk: Federated Learning with Zero-Knowledge Proofs (integrity-gated FL).

Branch v2: research demo → contribution rebuild. Agent kit: docs/agent-kit/.
Heavy imports are lazy so `import fedzk` / doctor do not require snarkjs.
"""

from __future__ import annotations

from typing import Any

__version__ = "1.2.0"

__all__ = [
    "__version__",
    "ZKProver",
    "ZKVerifier",
    "GradientQuantizer",
]


def __getattr__(name: str) -> Any:
    if name == "ZKProver":
        from fedzk.prover.zkgenerator import ZKProver

        return ZKProver
    if name == "ZKVerifier":
        from fedzk.prover.verifier import ZKVerifier

        return ZKVerifier
    if name == "GradientQuantizer":
        from fedzk.zk.input_normalization import GradientQuantizer

        return GradientQuantizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
