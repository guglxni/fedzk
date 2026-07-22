# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""ZK backend selection: snarkjs (default) | rust | auto.

FEDZK_ZK_BACKEND controls prove/verify engine. Rust path is fail-closed until
fedzk-zk arkworks verify returns 0 (Phase 3 U40).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class RustEngineUnavailable(RuntimeError):
    """Raised when engine=rust but binary missing or verify not wired."""


def selected_backend() -> str:
    return os.environ.get("FEDZK_ZK_BACKEND", "snarkjs").strip().lower() or "snarkjs"


def find_fedzk_zk() -> Optional[Path]:
    env = os.environ.get("FEDZK_ZK_BIN")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    which = shutil.which("fedzk-zk")
    if which:
        return Path(which)
    # Dev default: cargo workspace target under v2/rust/
    here = Path(__file__).resolve()
    root = here.parents[3]  # v2/
    for rel in (
        "rust/target/release/fedzk-zk",
        "rust/target/debug/fedzk-zk",
        "rust/fedzk-zk/target/release/fedzk-zk",
        "rust/fedzk-zk/target/debug/fedzk-zk",
    ):
        cand = root / rel
        if cand.is_file():
            return cand
    return None


def rust_health() -> Tuple[bool, Dict[str, Any]]:
    bin_path = find_fedzk_zk()
    if not bin_path:
        return False, {"ok": False, "error": "fedzk-zk binary not found"}
    try:
        out = subprocess.run(
            [str(bin_path), "health"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode != 0:
            return False, {"ok": False, "stderr": out.stderr}
        return True, json.loads(out.stdout)
    except (OSError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
        return False, {"ok": False, "error": str(e)}


def resolve_backend() -> str:
    choice = selected_backend()
    if choice == "auto":
        ok, _ = rust_health()
        return "rust" if ok else "snarkjs"
    if choice not in ("snarkjs", "rust", "auto"):
        raise ValueError(f"Unknown FEDZK_ZK_BACKEND={choice!r}")
    return choice


def verify_with_rust(
    proof: Dict[str, Any],
    public_inputs: List[Any],
    verification_key_path: str,
) -> bool:
    """Fail-closed: returns True only if fedzk-zk verify exits 0.

    Scaffold exits 3 (envelope OK, pairing TBD) → treated as unavailable.
    """
    bin_path = find_fedzk_zk()
    if not bin_path:
        raise RustEngineUnavailable(
            "FEDZK_ZK_BACKEND=rust but fedzk-zk not found. "
            "Build: cd v2/rust/fedzk-zk && cargo build --release"
        )

    with tempfile.TemporaryDirectory(prefix="fedzk-rust-") as td:
        tdp = Path(td)
        proof_path = tdp / "proof.json"
        public_path = tdp / "public.json"
        proof_path.write_text(json.dumps(proof))
        public_path.write_text(json.dumps(public_inputs))
        proc = subprocess.run(
            [
                str(bin_path),
                "verify",
                "--vkey",
                verification_key_path,
                "--proof",
                str(proof_path),
                "--public",
                str(public_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if proc.returncode == 0:
            return True
        if proc.returncode == 3:
            raise RustEngineUnavailable(
                "fedzk-zk envelope OK but arkworks pairing verify not wired yet "
                f"(stderr={proc.stderr.strip()!r})"
            )
        raise RustEngineUnavailable(
            f"fedzk-zk verify failed rc={proc.returncode}: {proc.stderr.strip()}"
        )
