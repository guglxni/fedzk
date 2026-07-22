#!/usr/bin/env python3
# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""Phase 4 attack-rejection suite — honest accept + tampered rejects.

Writes artifacts/transcripts/attack-rejection.json for paper Table metrics.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedzk.coordinator.api import app  # noqa: E402
from fedzk.prover.chunked import prove_update  # noqa: E402
from fedzk.zk.chunk_protocol import commit_quantized  # noqa: E402
from fedzk.zk.input_normalization import GradientQuantizer  # noqa: E402


def _reset():
    from fedzk.coordinator import logic as logic_mod

    logic_mod.pending_updates.clear()
    logic_mod.security_manager.client_request_times.clear()
    logic_mod.security_manager.failed_verifications.clear()
    logic_mod.security_manager.blocked_clients.clear()


def _honest_bundle(grads: Dict[str, torch.Tensor], n: int = 4) -> Dict[str, Any]:
    proved = prove_update(grads, n=n)
    if proved.get("mode") == "chunked":
        return {
            "proved": proved,
            "bundle": {
                "commitment": proved["commitment"],
                "n": proved["n"],
                "circuit_id": proved["circuit_id"],
                "chunks": proved["chunks"],
            },
        }
    q, _ = GradientQuantizer(scale_factor=1000).quantize_gradients(grads)
    flat: List[int] = []
    for vals in q.values():
        flat.extend(int(v) for v in vals)
    commitment = proved.get("commitment") or commit_quantized(flat)
    return {
        "proved": proved,
        "bundle": {
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
        },
    }


def _submit(client: TestClient, grads: Dict[str, torch.Tensor], bundle: dict, cid: str):
    grad_lists = {k: v.detach().cpu().flatten().tolist() for k, v in grads.items()}
    return client.post(
        "/submit_update",
        json={
            "gradients": grad_lists,
            "proof": {},
            "public_inputs": [],
            "client_id": cid,
            "chunk_bundle": bundle,
        },
    )


AttackMutator = Callable[[dict], dict]


def mutate_pi_a(bundle: dict) -> dict:
    b = copy.deepcopy(bundle)
    a = b["chunks"][0]["proof"]["pi_a"]
    # Flip a limb — invalidates pairing
    a[0] = str((int(a[0]) + 1) % (2**255))
    return b


def mutate_public(bundle: dict) -> dict:
    b = copy.deepcopy(bundle)
    pubs = list(b["chunks"][0]["public_inputs"])
    pubs[0] = str(int(pubs[0]) + 7)
    b["chunks"][0]["public_inputs"] = pubs
    return b


def mutate_commitment_top(bundle: dict) -> dict:
    """Tamper only outer commitment — meta stays honest → structural reject."""
    b = copy.deepcopy(bundle)
    b["commitment"] = "deadbeef" * 8
    return b


def mutate_commitment_meta(bundle: dict) -> dict:
    """Tamper only chunk meta commitment — outer stays honest → structural reject."""
    b = copy.deepcopy(bundle)
    if b["chunks"][0].get("meta"):
        b["chunks"][0]["meta"]["commitment"] = "cafebabe" * 8
    return b


def mutate_empty_proof(bundle: dict) -> dict:
    b = copy.deepcopy(bundle)
    b["chunks"][0]["proof"] = {}
    return b


def run_suite(n: int = 4) -> dict:
    _reset()
    client = TestClient(app)
    grads = {"w": torch.randn(n) * 0.001}
    honest = _honest_bundle(grads, n=n)

    cases: List[Dict[str, Any]] = []

    # Honest — expect accept
    r = _submit(client, grads, honest["bundle"], "atk-honest")
    cases.append(
        {
            "case": "honest",
            "expect": "accept",
            "http": r.status_code,
            "ok": r.status_code == 200,
            "body": r.json() if r.status_code < 500 else {"error": r.text[:200]},
        }
    )

    mutators = [
        ("tamper_pi_a", mutate_pi_a, "reject"),
        ("tamper_public", mutate_public, "reject"),
        ("tamper_commitment_top", mutate_commitment_top, "reject"),
        ("tamper_commitment_meta", mutate_commitment_meta, "reject"),
        ("empty_proof", mutate_empty_proof, "reject"),
    ]
    for name, fn, expect in mutators:
        _reset()
        # re-prove so each attack is independent of rate-limit state from prior fails
        pack = _honest_bundle(grads, n=n)
        bad = fn(pack["bundle"])
        r = _submit(client, grads, bad, f"atk-{name}")
        rejected = r.status_code >= 400
        cases.append(
            {
                "case": name,
                "expect": expect,
                "http": r.status_code,
                "ok": rejected if expect == "reject" else r.status_code == 200,
                "body": r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text[:200]},
            }
        )

    accepted = sum(1 for c in cases if c["case"] == "honest" and c["ok"])
    rejected = sum(1 for c in cases if c["expect"] == "reject" and c["ok"])
    attack_n = sum(1 for c in cases if c["expect"] == "reject")
    out = {
        "wire": "fedzk.proof.v1",
        "experiment": "attack-rejection",
        "n_circuit": n,
        "cases": cases,
        "metrics": {
            "honest_accept": accepted == 1,
            "attacks_total": attack_n,
            "attacks_rejected": rejected,
            "rejection_rate": (rejected / attack_n) if attack_n else 0.0,
        },
    }
    dest = ROOT / "artifacts" / "transcripts" / "attack-rejection.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({"wrote": str(dest), "metrics": out["metrics"]}, indent=2))
    return out


if __name__ == "__main__":
    run_suite()
