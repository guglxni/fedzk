#!/usr/bin/env python3
# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""Phase 4 measurement harness — Adult LR FEDzk timings (synthetic).

Config: spec/experiments/adult-lr-fedzk.yaml (PyYAML optional).
Output: artifacts/transcripts/adult-lr-measure.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedzk.client.trainer import LocalTrainer  # noqa: E402
from fedzk.coordinator.api import app  # noqa: E402
from fedzk.prover.chunked import prove_update  # noqa: E402
from fedzk.prover.engine import find_fedzk_zk, verify_with_rust  # noqa: E402
from fedzk.prover.zkgenerator import ASSET_DIR  # noqa: E402
from fedzk.zk.circuit_config import profile_for_n  # noqa: E402
from fedzk.zk.chunk_protocol import commit_quantized  # noqa: E402
from fedzk.zk.input_normalization import GradientQuantizer  # noqa: E402


def _load_cfg() -> dict:
    path = ROOT / "spec" / "experiments" / "adult-lr-fedzk.yaml"
    defaults = {
        "federated": {"clients": 3, "rounds": 1},
        "zk": {"n_circuit": 64},
        "model": {"learning_rate": 0.05, "epochs_per_round": 1},
        "data": {"n_samples": 64, "n_features": 8},
    }
    if not path.is_file():
        return defaults
    try:
        import yaml  # type: ignore

        raw = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return defaults
    out = defaults
    for k, v in raw.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def _make_data(n: int, d: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w = torch.randn(d, generator=g)
    y = ((X @ w) > 0).long()
    return X, y


def _bundle(proved: dict, grads: dict) -> dict:
    if proved.get("mode") == "chunked":
        return {
            "commitment": proved["commitment"],
            "n": proved["n"],
            "circuit_id": proved["circuit_id"],
            "chunks": proved["chunks"],
        }
    q, _ = GradientQuantizer(scale_factor=1000).quantize_gradients(grads)
    flat = []
    for vals in q.values():
        flat.extend(int(v) for v in vals)
    commitment = proved.get("commitment") or commit_quantized(flat)
    return {
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


def _reset():
    from fedzk.coordinator import logic as logic_mod

    logic_mod.pending_updates.clear()
    logic_mod.security_manager.client_request_times.clear()
    logic_mod.security_manager.failed_verifications.clear()
    logic_mod.security_manager.blocked_clients.clear()


def _stat(xs: list[float]) -> dict:
    if not xs:
        return {}
    xs_sorted = sorted(xs)
    return {
        "n": len(xs),
        "mean_ms": statistics.fmean(xs),
        "p50_ms": xs_sorted[len(xs_sorted) // 2],
        "max_ms": max(xs),
    }


def run_baseline(cfg: dict) -> dict:
    """Same train loop without prove/verify/submit — wall-time baseline arm."""
    clients = int(cfg["federated"]["clients"])
    rounds = int(cfg["federated"]["rounds"])
    n_samples = int(cfg["data"]["n_samples"])
    n_features = int(cfg["data"]["n_features"])
    lr = float(cfg["model"]["learning_rate"])
    epochs = int(cfg["model"]["epochs_per_round"])
    train_ms: list[float] = []
    losses: list[float] = []
    for r in range(rounds):
        for c in range(clients):
            X, y = _make_data(n_samples, n_features, seed=r * 100 + c)
            trainer = LocalTrainer(
                model_type="linear", learning_rate=lr, device="cpu", hidden_size=4
            )
            trainer.input_size = n_features
            trainer.model = trainer.create_model(n_features, num_classes=2).to(
                trainer.device
            )
            trainer.optimizer = trainer.create_optimizer(trainer.model)
            trainer.criterion = torch.nn.CrossEntropyLoss()
            trainer.dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X, y), batch_size=16, shuffle=True
            )
            t0 = time.perf_counter()
            metrics = trainer.train(epochs=epochs)
            train_ms.append((time.perf_counter() - t0) * 1000.0)
            losses.append(float(metrics.get("loss", 0.0) or 0.0))
    return {
        "arm": "baseline_no_zk",
        "metrics": {
            "train": _stat(train_ms),
            "train_loss_mean": statistics.fmean(losses) if losses else None,
        },
    }


def run_measure() -> dict:
    cfg = _load_cfg()
    clients = int(cfg["federated"]["clients"])
    rounds = int(cfg["federated"]["rounds"])
    n_circuit = int(cfg["zk"]["n_circuit"])
    n_samples = int(cfg["data"]["n_samples"])
    n_features = int(cfg["data"]["n_features"])
    lr = float(cfg["model"]["learning_rate"])
    epochs = int(cfg["model"]["epochs_per_round"])

    baseline = run_baseline(cfg)

    _reset()
    http = TestClient(app)
    profile = profile_for_n(n_circuit)
    vkey = str(ASSET_DIR / profile.vkey_name)
    rust_ready = find_fedzk_zk() is not None

    prove_ms: list[float] = []
    rust_ms: list[float] = []
    submit_ms: list[float] = []
    losses: list[float] = []
    round_rows = []

    for r in range(rounds):
        events: dict = {"round": r, "clients": []}
        for c in range(clients):
            X, y = _make_data(n_samples, n_features, seed=r * 100 + c)
            trainer = LocalTrainer(
                model_type="linear", learning_rate=lr, device="cpu", hidden_size=4
            )
            trainer.input_size = n_features
            trainer.model = trainer.create_model(n_features, num_classes=2).to(
                trainer.device
            )
            trainer.optimizer = trainer.create_optimizer(trainer.model)
            trainer.criterion = torch.nn.CrossEntropyLoss()
            trainer.dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X, y), batch_size=16, shuffle=True
            )
            metrics = trainer.train(epochs=epochs)
            loss = float(metrics.get("loss", 0.0) or 0.0)
            losses.append(loss)
            grads = trainer.get_gradients()

            t0 = time.perf_counter()
            proved = prove_update(grads, n=n_circuit)
            prove_ms.append((time.perf_counter() - t0) * 1000.0)

            rust_ok = None
            if rust_ready and proved.get("mode") == "single":
                t1 = time.perf_counter()
                rust_ok = verify_with_rust(
                    proved["proof"], proved["public_inputs"], vkey
                )
                rust_ms.append((time.perf_counter() - t1) * 1000.0)

            grad_lists = {
                k: v.detach().cpu().flatten().tolist() for k, v in grads.items()
            }
            payload = {
                "gradients": grad_lists,
                "proof": {},
                "public_inputs": [],
                "client_id": f"measure-c{c}",
                "chunk_bundle": _bundle(proved, grads),
            }
            t2 = time.perf_counter()
            resp = http.post("/submit_update", json=payload)
            submit_ms.append((time.perf_counter() - t2) * 1000.0)
            if resp.status_code >= 400:
                raise RuntimeError(f"submit failed: {resp.status_code} {resp.text}")
            events["clients"].append(
                {
                    "client_id": f"measure-c{c}",
                    "http": resp.status_code,
                    "prove_mode": proved.get("mode"),
                    "train_loss": loss,
                    "rust_verify": rust_ok,
                    "body": resp.json(),
                }
            )
        events["status"] = http.get("/status").json()
        round_rows.append(events)

    out = {
        "wire": "fedzk.proof.v1",
        "experiment": "adult-lr-measure",
        "n_circuit": n_circuit,
        "ceremony": profile.ceremony,
        "baseline": baseline,
        "rounds": round_rows,
        "metrics": {
            "prove": _stat(prove_ms),
            "rust_verify": _stat(rust_ms),
            "submit": _stat(submit_ms),
            "train_loss_mean": statistics.fmean(losses) if losses else None,
        },
    }
    dest = ROOT / "artifacts" / "transcripts" / "adult-lr-measure.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {
                "wrote": str(dest),
                "baseline": baseline["metrics"],
                "fedzk": out["metrics"],
            },
            indent=2,
        )
    )
    return out


if __name__ == "__main__":
    run_measure()
