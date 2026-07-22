# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
Environment / toolchain doctor for FEDzk (Bolt 1).

Reports readiness without generating proofs. Prefer this over trusting README claims.
"""

from __future__ import annotations

import hashlib
import importlib
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ASSET_DIR = Path(__file__).resolve().parent / "zk"

# Canonical N_dev artifacts (standard + secure)
CORE_ARTIFACTS = (
    "model_update.wasm",
    "proving_key.zkey",
    "verification_key.json",
    "model_update_secure.wasm",
    "proving_key_secure.zkey",
    "verification_key_secure.json",
)


def _find_artifact_manifest() -> Optional[Path]:
    """Locate freeze pin: v2/artifacts/artifact-manifest.json (preferred) or beside package."""
    candidates = [
        Path(__file__).resolve().parents[2] / "artifacts" / "artifact-manifest.json",  # v2/artifacts
        Path(__file__).resolve().parents[3] / "artifacts" / "artifact-manifest.json",
        ASSET_DIR / "artifact-manifest.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_import(name: str) -> Tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", getattr(mod, "VERSION", "?"))
        return True, str(ver)
    except Exception as exc:  # noqa: BLE001 — doctor must never crash mid-report
        return False, str(exc)


def run_doctor() -> Dict[str, Any]:
    """
    Collect doctor findings.

    Returns:
        Dict with overall_ok (bool), checks (list), errors/warnings.
    """
    checks: List[Dict[str, Any]] = []
    errors: List[str] = []
    warnings: List[str] = []

    # 1) CLI entrypoint must resolve (guard against re-introducing cli/ package shadow)
    try:
        from fedzk.cli import app as cli_app  # type: ignore
        import fedzk.cli as cli_mod

        origin = getattr(cli_mod, "__file__", "") or ""
        shadow = origin.replace("\\", "/").endswith("/cli/__init__.py")
        if shadow or cli_app is None:
            checks.append(
                {
                    "id": "cli_module",
                    "ok": False,
                    "detail": f"fedzk.cli resolves to package shadow: {origin}",
                }
            )
            errors.append("CLI package shadow detected; remove src/fedzk/cli/ directory")
        else:
            checks.append(
                {
                    "id": "cli_module",
                    "ok": True,
                    "detail": f"fedzk.cli → {origin}",
                }
            )
    except Exception as exc:  # noqa: BLE001
        checks.append({"id": "cli_module", "ok": False, "detail": str(exc)})
        errors.append(f"Cannot import fedzk.cli:app — {exc}")

    # 2) Core Python deps
    for pkg in ("numpy", "typer", "fastapi", "torch"):
        ok, detail = _check_import(pkg)
        checks.append({"id": f"dep_{pkg}", "ok": ok, "detail": detail})
        if not ok:
            errors.append(f"Missing dependency {pkg}: {detail}")

    # 3) Toolchain binaries
    for bin_name in ("snarkjs", "node"):
        path = shutil.which(bin_name)
        ok = path is not None
        checks.append({"id": f"bin_{bin_name}", "ok": ok, "detail": path or "not found"})
        if not ok:
            errors.append(f"{bin_name} not on PATH")

    circom = shutil.which("circom")
    checks.append(
        {
            "id": "bin_circom",
            "ok": circom is not None,
            "detail": circom or "not found (optional for prove if wasm present)",
        }
    )
    if circom is None:
        warnings.append("circom not on PATH — cannot rebuild circuits; prove may still work with shipped wasm")

    # 4) Artifacts + hashes
    artifact_hashes: Dict[str, str] = {}
    for name in CORE_ARTIFACTS:
        path = ASSET_DIR / name
        if not path.is_file():
            checks.append({"id": f"artifact_{name}", "ok": False, "detail": f"missing: {path}"})
            errors.append(f"Missing ZK artifact: {path}")
        else:
            digest = _sha256_file(path)
            artifact_hashes[name] = digest
            checks.append(
                {
                    "id": f"artifact_{name}",
                    "ok": True,
                    "detail": f"{path} sha256={digest[:16]}…",
                }
            )

    # 4b) Freeze pin vs artifacts/artifact-manifest.json
    manifest_path = _find_artifact_manifest()
    if manifest_path is None:
        checks.append(
            {
                "id": "artifact_manifest",
                "ok": False,
                "detail": "artifact-manifest.json not found (expected v2/artifacts/)",
            }
        )
        errors.append("Missing artifacts/artifact-manifest.json freeze pin")
    else:
        try:
            import json

            manifest = json.loads(manifest_path.read_text())
            pinned = manifest.get("artifacts") or {}
            mismatches = []
            for name, expected in pinned.items():
                actual = artifact_hashes.get(name)
                if actual is None:
                    mismatches.append(f"{name}: missing on disk")
                elif actual != expected:
                    mismatches.append(f"{name}: pin drift")
            ok = len(mismatches) == 0
            checks.append(
                {
                    "id": "artifact_manifest",
                    "ok": ok,
                    "detail": str(manifest_path) if ok else "; ".join(mismatches),
                }
            )
            if not ok:
                errors.extend(mismatches)
            if manifest.get("policy") == "freeze":
                checks.append(
                    {
                        "id": "artifact_policy",
                        "ok": True,
                        "detail": f"policy=freeze n_dev={manifest.get('n_dev')} frozen_at={manifest.get('frozen_at')}",
                    }
                )
        except Exception as exc:  # noqa: BLE001
            checks.append({"id": "artifact_manifest", "ok": False, "detail": str(exc)})
            errors.append(f"Failed reading artifact manifest: {exc}")

    # 5) ZKValidator (if available)
    try:
        from fedzk.prover.zk_validator import ZKValidator

        validator = ZKValidator(str(ASSET_DIR))
        validation = validator.validate_toolchain()
        status = validation.get("overall_status", "unknown")
        ok = status != "failed"
        checks.append({"id": "zk_validator", "ok": ok, "detail": status})
        if not ok:
            for err in validation.get("errors", []):
                errors.append(str(err))
        for warn in validation.get("warnings", []):
            warnings.append(str(warn))
    except Exception as exc:  # noqa: BLE001
        checks.append({"id": "zk_validator", "ok": False, "detail": str(exc)})
        warnings.append(f"ZKValidator unavailable: {exc}")

    # 6) Honesty: N_dev capacity
    checks.append(
        {
            "id": "circuit_n_dev",
            "ok": True,
            "detail": "model_update circuit fixed N_dev=4; pad/truncate is logged (Phase0)",
        }
    )
    warnings.append("Circuit capacity N_dev=4 — not production model size; see PLAN Phase 1")

    overall_ok = len(errors) == 0
    return {
        "overall_ok": overall_ok,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "artifact_hashes": artifact_hashes,
        "asset_dir": str(ASSET_DIR),
        "python": sys.version.split()[0],
    }


def format_doctor_report(report: Dict[str, Any]) -> str:
    lines = [
        "FEDzk doctor",
        f"  python: {report.get('python')}",
        f"  asset_dir: {report.get('asset_dir')}",
        f"  overall: {'OK' if report.get('overall_ok') else 'FAIL'}",
        "",
        "Checks:",
    ]
    for c in report.get("checks", []):
        mark = "✓" if c.get("ok") else "✗"
        lines.append(f"  {mark} [{c.get('id')}] {c.get('detail')}")
    if report.get("errors"):
        lines.append("")
        lines.append("Errors:")
        for e in report["errors"]:
            lines.append(f"  • {e}")
    if report.get("warnings"):
        lines.append("")
        lines.append("Warnings:")
        for w in report["warnings"]:
            lines.append(f"  • {w}")
    return "\n".join(lines)
