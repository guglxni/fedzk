#!/usr/bin/env python3
# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""Adult-style LR smoke: synthetic binary data → LocalTrainer → prove → coordinator FedAvg.

Runs entirely under v2/ with real symbols (no fictional APIs).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from fastapi.testclient import TestClient

# Ensure v2 package is importable when run as script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedzk.client.trainer import LocalTrainer  # noqa: E402
from fedzk.coordinator.api import app  # noqa: E402
from fedzk.prover.chunked import prove_update  # noqa: E402


def _make_synthetic_adult(n: int = 64, d: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w = torch.randn(d, generator=g)
    logits = X @ w
    y = (logits > 0).long()
    return X, y


def run_smoke(rounds: int = 1, clients: int = 3, n_circuit: int = 64) -> dict:
    # Fresh coordinator state for reproducible smoke (loop-friendly)
    from fedzk.coordinator import logic as logic_mod

    logic_mod.pending_updates.clear()
    logic_mod.security_manager.client_request_times.clear()
    logic_mod.security_manager.failed_verifications.clear()
    logic_mod.security_manager.blocked_clients.clear()

    client = TestClient(app)
    transcript = {"rounds": [], "wire": "fedzk.proof.v1", "n_circuit": n_circuit}

    for r in range(rounds):
        round_events = {"round": r, "clients": []}
        for c in range(clients):
            X, y = _make_synthetic_adult(seed=r * 100 + c)
            # Tiny linear model: input_size=d, output=2
            trainer = LocalTrainer(
                model_type="linear",
                learning_rate=0.05,
                device="cpu",
                hidden_size=4,
            )
            # Bypass file load — inject tensors
            trainer.input_size = X.shape[1]
            trainer.model = trainer.create_model(X.shape[1], num_classes=2).to(trainer.device)
            trainer.optimizer = trainer.create_optimizer(trainer.model)
            trainer.criterion = torch.nn.CrossEntropyLoss()
            ds = torch.utils.data.TensorDataset(X, y)
            trainer.dataloader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)

            metrics = trainer.train(epochs=1)
            grads = trainer.get_gradients()
            # Prove (chunked if |θ| > n_circuit)
            proved = prove_update(grads, n=n_circuit)
            # Gradients as lists for API
            grad_lists = {k: v.detach().cpu().flatten().tolist() for k, v in grads.items()}

            # Coordinator chunk path uses snarkjs verify (fail-closed, no attack-heuristic false rejects).
            # Normalize single proofs into a 1-chunk bundle.
            if proved.get("mode") == "chunked":
                bundle = {
                    "commitment": proved["commitment"],
                    "n": proved["n"],
                    "circuit_id": proved["circuit_id"],
                    "chunks": proved["chunks"],
                }
            else:
                from fedzk.zk.chunk_protocol import commit_quantized
                from fedzk.zk.input_normalization import GradientQuantizer

                q, _ = GradientQuantizer(scale_factor=1000).quantize_gradients(grads)
                flat = []
                for vals in q.values():
                    flat.extend(int(v) for v in vals)
                commitment = proved.get("commitment") or commit_quantized(flat)
                bundle = {
                    "commitment": commitment,
                    "n": proved["n"],
                    "circuit_id": proved["circuit_id"],
                    "chunks": [
                        {
                            "meta": {
                                "circuit_id": proved["circuit_id"],
                                "n": proved["n"],
                                "chunk_index": 0,
                                "chunk_count": 1,
                                "commitment": commitment,
                                "wire": "fedzk.proof.v1",
                            },
                            "proof": proved["proof"],
                            "public_inputs": proved["public_inputs"],
                        }
                    ],
                }

            payload = {
                "gradients": grad_lists,
                "proof": {},
                "public_inputs": [],
                "client_id": f"c{c}",
                "chunk_bundle": bundle,
            }

            resp = client.post("/submit_update", json=payload)
            event = {
                "client_id": f"c{c}",
                "http": resp.status_code,
                "body": resp.json() if resp.status_code < 500 else {"error": resp.text},
                "prove_mode": proved.get("mode"),
                "train_loss": metrics.get("loss"),
            }
            round_events["clients"].append(event)
            if resp.status_code >= 400:
                raise RuntimeError(f"submit failed: {event}")

        status = client.get("/status").json()
        round_events["status"] = status
        transcript["rounds"].append(round_events)

    out = ROOT / "artifacts" / "transcripts" / "adult-lr-smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(transcript, indent=2))
    return transcript


if __name__ == "__main__":
    t = run_smoke()
    last = t["rounds"][-1]
    print(json.dumps({"rounds": len(t["rounds"]), "last_status": last["status"]}, indent=2))
    print("wrote", ROOT / "artifacts" / "transcripts" / "adult-lr-smoke.json")
