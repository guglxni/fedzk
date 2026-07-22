# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""Optional Flower adapter spike (Phase 2) — no Flower import required at install.

Usage (when flwr is installed):
    from fedzk.adapters.flower import FEDzkClient, FEDzkStrategy
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from fedzk.prover.chunked import prove_update
from fedzk.zk.circuit_config import require_supported_n


class FEDzkClientMixin:
    """Mixin that wraps local grads → prove_update before sending to server.

    Does not subclass flwr.client.NumPyClient so the package stays Flower-optional.
    Callers compose this with their Flower client.
    """

    def __init__(self, n_circuit: int = 64, prefer_single_if_fits: bool = True):
        require_supported_n(n_circuit)
        self.n_circuit = n_circuit
        self.prefer_single_if_fits = prefer_single_if_fits

    def prove_numpy_grads(
        self, named_grads: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        return prove_update(
            named_grads,
            n=self.n_circuit,
            prefer_single_if_fits=self.prefer_single_if_fits,
        )

    def pack_fit_metrics(self, proved: Dict[str, Any]) -> Dict[str, Any]:
        """Metrics dict safe to attach to Flower fit return (JSON-serializable)."""
        if proved.get("mode") == "chunked":
            return {
                "fedzk_mode": "chunked",
                "fedzk_n": proved["n"],
                "fedzk_circuit_id": proved["circuit_id"],
                "fedzk_commitment": proved["commitment"],
                "fedzk_num_chunks": len(proved.get("chunks", [])),
            }
        return {
            "fedzk_mode": "single",
            "fedzk_n": proved.get("n"),
            "fedzk_circuit_id": proved.get("circuit_id"),
        }


def ndarray_list_to_named(
    arrays: List[Any], names: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """Convert Flower-style NDArray list into named torch tensors for prove_update."""
    out: Dict[str, torch.Tensor] = {}
    for i, arr in enumerate(arrays):
        key = names[i] if names and i < len(names) else f"p{i}"
        out[key] = torch.as_tensor(arr, dtype=torch.float32)
    return out


def try_import_flwr() -> Tuple[bool, Optional[Any]]:
    try:
        import flwr  # type: ignore

        return True, flwr
    except ImportError:
        return False, None
