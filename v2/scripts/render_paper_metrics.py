#!/usr/bin/env python3
# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""Render paper-facing metrics tables from Phase4 transcripts → Markdown."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANS = ROOT / "artifacts" / "transcripts"
OUT = ROOT / "artifacts" / "paper" / "metrics-tables.md"


def _row_ms(label: str, block: dict) -> str:
    if not block:
        return f"| {label} | — | — | — |"
    return (
        f"| {label} | {block.get('mean_ms', 0):.1f} | "
        f"{block.get('p50_ms', 0):.1f} | {block.get('max_ms', 0):.1f} |"
    )


def main() -> None:
    measure = json.loads((TRANS / "adult-lr-measure.json").read_text())
    attack_path = TRANS / "attack-rejection.json"
    attack = json.loads(attack_path.read_text()) if attack_path.is_file() else None

    lines = [
        "# FEDzk Phase 4 — metrics tables (auto-generated)",
        "",
        f"Source: `adult-lr-measure.json` (n_circuit={measure.get('n_circuit')}, "
        f"ceremony={measure.get('ceremony')}).",
        "",
        "## T1 — Adult LR wall-time (ms / client update)",
        "",
        "| Arm | mean | p50 | max |",
        "| --- | ---: | ---: | ---: |",
    ]
    base = (measure.get("baseline") or {}).get("metrics") or {}
    lines.append(_row_ms("baseline train (no ZK)", base.get("train") or {}))
    m = measure.get("metrics") or {}
    lines.append(_row_ms("FEDzk prove (snarkjs)", m.get("prove") or {}))
    lines.append(_row_ms("FEDzk rust verify", m.get("rust_verify") or {}))
    lines.append(_row_ms("FEDzk coordinator submit", m.get("submit") or {}))
    lines.extend(["", f"Train loss (FEDzk arm mean): `{m.get('train_loss_mean')}`", ""])

    if attack:
        met = attack.get("metrics") or {}
        lines.extend(
            [
                "## T3 — Attack rejection (coordinator fail-closed)",
                "",
                f"- Honest accept: `{met.get('honest_accept')}`",
                f"- Attacks rejected: `{met.get('attacks_rejected')}/{met.get('attacks_total')}`",
                f"- Rejection rate: `{met.get('rejection_rate')}`",
                "",
                "| Case | expect | http | ok |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for c in attack.get("cases") or []:
            lines.append(
                f"| {c.get('case')} | {c.get('expect')} | {c.get('http')} | {c.get('ok')} |"
            )
        lines.append("")

    lines.extend(
        [
            "---",
            "Regenerate: `python scripts/adult_lr_measure.py && "
            "python scripts/attack_rejection_suite.py && "
            "python scripts/render_paper_metrics.py`",
        ]
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
