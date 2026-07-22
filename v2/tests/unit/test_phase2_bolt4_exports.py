# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""Bolt 4 — SDK exports + Adult smoke transcript schema."""

from __future__ import annotations

import json
from pathlib import Path

import fedzk
from jsonschema import validate

ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPT = ROOT / "artifacts" / "transcripts" / "adult-lr-smoke.json"
SCHEMA = ROOT / "artifacts" / "transcripts" / "adult-lr-smoke.schema.json"


def test_package_exports_phase1_api():
    assert fedzk.__version__ == "1.2.0"
    assert callable(fedzk.ZKProver)
    assert callable(fedzk.ZKVerifier)
    assert callable(fedzk.GradientQuantizer)
    assert callable(fedzk.prove_update)
    assert 256 in fedzk.SHIPPED_N
    assert fedzk.profile_for_n(256).n == 256
    assert fedzk.profile_for_n(256).ceremony == "dev_unsafe"


def test_adult_lr_smoke_transcript_matches_schema():
    assert TRANSCRIPT.is_file(), "run scripts/adult_lr_smoke.py first"
    assert SCHEMA.is_file()
    data = json.loads(TRANSCRIPT.read_text())
    schema = json.loads(SCHEMA.read_text())
    validate(instance=data, schema=schema)
    assert data["n_circuit"] in (4, 64, 256)
    assert data["wire"] == "fedzk.proof.v1"
