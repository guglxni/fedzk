#!/usr/bin/env bash
# Local CI mirror for isolated v2/ tree (GitHub Actions root workflows cannot live here).
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== fedzk doctor =="
.venv/bin/fedzk doctor
echo "== pytest unit =="
.venv/bin/pytest tests/unit/ -q
echo "== cargo build fedzk-zk =="
(cd rust && cargo build -p fedzk-zk)
./rust/target/debug/fedzk-zk health
echo "CI_LOCAL_OK"
echo "== e5 transcript present =="
test -f artifacts/transcripts/e5-backend-ablation.json
echo "== attack transcript present =="
test -f artifacts/transcripts/attack-rejection.json
