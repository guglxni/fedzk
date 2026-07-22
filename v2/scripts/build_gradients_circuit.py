#!/usr/bin/env python3
"""Build ModelUpdateGradients(N) with a DEV (unsafe) Groth16 ceremony.

Usage:
  python scripts/build_gradients_circuit.py --n 64
  python scripts/build_gradients_circuit.py --n 256

Does NOT replace frozen N=4 legacy artifacts. Labels ceremony=dev_unsafe in output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CIRCOM_TEMPLATE = ROOT / "src/fedzk/zk/circuits/model_update_gradients.circom"
ASSET = ROOT / "src/fedzk/zk"
MANIFEST = ROOT / "artifacts/artifact-manifest.json"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, choices=[4, 64, 256])
    ap.add_argument("--ptau-power", type=int, default=14, help="powersoftau size (12 ok for n=64)")
    args = ap.parse_args()
    n = args.n
    if n == 256 and args.ptau_power < 14:
        args.ptau_power = 16

    src = CIRCOM_TEMPLATE.read_text().replace(
        "component main = ModelUpdateGradients(4);",
        f"component main = ModelUpdateGradients({n});",
    )
    circom_path = ROOT / "src/fedzk/zk/circuits" / f"model_update_gradients_n{n}.circom"
    circom_path.write_text(src)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        subprocess.check_call(
            ["circom", str(circom_path), "--r1cs", "--wasm", "-o", str(td)]
        )
        r1cs = td / "model_update_gradients.r1cs"
        # circom names output from file stem
        stem = circom_path.stem
        r1cs = td / f"{stem}.r1cs"
        wasm = td / f"{stem}_js" / f"{stem}.wasm"
        ptau0 = td / "pot_0000.ptau"
        ptau1 = td / "pot_0001.ptau"
        ptau_f = td / "pot_final.ptau"
        z0 = td / "z_0000.zkey"
        zf = td / "z_final.zkey"
        vkey = td / "vkey.json"

        subprocess.check_call(
            ["snarkjs", "powersoftau", "new", "bn128", str(args.ptau_power), str(ptau0)]
        )
        subprocess.check_call(
            [
                "snarkjs",
                "powersoftau",
                "contribute",
                str(ptau0),
                str(ptau1),
                "--name=fedzk-v2-dev",
                f"-e=fedzk-dev-{n}",
            ]
        )
        subprocess.check_call(
            ["snarkjs", "powersoftau", "prepare", "phase2", str(ptau1), str(ptau_f)]
        )
        subprocess.check_call(
            ["snarkjs", "groth16", "setup", str(r1cs), str(ptau_f), str(z0)]
        )
        subprocess.check_call(
            [
                "snarkjs",
                "zkey",
                "contribute",
                str(z0),
                str(zf),
                "--name=fedzk-v2-dev",
                f"-e=fedzk-zkey-{n}",
            ]
        )
        subprocess.check_call(
            ["snarkjs", "zkey", "export", "verificationkey", str(zf), str(vkey)]
        )

        out_wasm = ASSET / f"model_update_gradients_n{n}.wasm"
        out_zkey = ASSET / f"proving_key_gradients_n{n}.zkey"
        out_vkey = ASSET / f"verification_key_gradients_n{n}.json"
        shutil.copy2(wasm, out_wasm)
        shutil.copy2(zf, out_zkey)
        shutil.copy2(vkey, out_vkey)
        hashes = {
            out_wasm.name: sha256(out_wasm),
            out_zkey.name: sha256(out_zkey),
            out_vkey.name: sha256(out_vkey),
        }
        print(json.dumps({"n": n, "ceremony": "dev_unsafe", "artifacts": hashes}, indent=2))

        if MANIFEST.is_file():
            man = json.loads(MANIFEST.read_text())
            man.setdefault("artifacts", {}).update(hashes)
            man.setdefault("artifacts_n64_dev" if n == 64 else f"artifacts_n{n}_dev", {}).update(hashes)
            shipped = set(man.get("n_shipped") or [4])
            shipped.add(n)
            man["n_shipped"] = sorted(shipped)
            MANIFEST.write_text(json.dumps(man, indent=2) + "\n")
            print("updated", MANIFEST)


if __name__ == "__main__":
    main()
