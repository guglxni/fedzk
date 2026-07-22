# Contribution: aidlc-quality-agent

**Intent:** `260722-fedzk-contribution`  
**Stage:** Inception / practices-discovery  
**Scope:** Construction Bolt 1 · Phase 0 hygiene  
**Depth:** Comprehensive for crypto (per AIDLC compose + `project.md`)  
**Traces:** `docs/agent-kit/QUALITY.html`, `REQUIREMENTS.html` (FR-CLI-01, FR-DEP-01, FR-Q-01, FR-ART-01, FR-API-01; NFR-C-01, NFR-S-01, NFR-D-01; SEC-01/05), `PLAN.html` Phase 0 exit gate, `aidlc/.../construction.md`

---

## Test strategy level (Comprehensive for crypto)

Bolt 1 / Phase 0 uses **Comprehensive** depth on the crypto walking path and **Standard** elsewhere.

| Layer | Bolt 1 expectation | Maps to |
| --- | --- | --- |
| Unit | Quantizer, prover/verifier wiring, coordinator verify-before-aggregate, CLI resolution | FR-Q-01, FR-CLI-01, FR-CRD-01 (skeleton), NFR-C-01 |
| Golden ZK | Fixed vectors prove + verify; **no mock proofs on default path** | QUALITY §1 Crypto; construction.md |
| Integration / walking skeleton | `quantize → prove → verify → FedAvg` on `N≥4` (`N_dev` OK if still 4) | PLAN Phase 0 exit; construction.md |
| Security hygiene | Secrets untracked; gitleaks clean; bandit high=0 on touched paths | NFR-S-01, SEC-05; QUALITY §1 Security |
| Docs / API honesty | Documented imports resolve; getting-started commands smoke | FR-API-01, NFR-D-01; QUALITY honesty policy |
| Deferred | Full compliance suites, Helm, Flower, Rust cross-verify, N≥64 circuits | Phase 1–5; compliance agent deferred until crypto green |

**Coverage bar (Bolt 1, enforced vs aspirational):**

- **Enforced:** critical path unit+golden green; no silent skips of crypto asserts.
- **Target (NFR-C-01):** ≥90% line coverage on `prover` / `coordinator` / `quantizer` modules touched by the skeleton — track in CI; do not invent coverage theater on compliance packages.
- **Aspirational (do not gate Bolt 1):** `quality-gates/config.yaml` prove-latency &lt;2s — reconcile later per QUALITY §3 (`enforced:` vs `aspirational:`).

**Forbidden for Bolt 1 “green”:**

- Mock / stub proofs as the default prove path.
- Tests importing fictional APIs (`CoordinatorAPI`, `ZKGenerator`, etc. — AIDLC sensor).
- README / badges claiming “Production Ready”, “100% Tested”, “Enterprise Deployed” (QUALITY §6).

---

## Sensors (commands that must pass)

Deterministic checks for Bolt 1 / Phase 0. A sensor fails closed: exit ≠ 0 blocks gate approval.

### S0 — Artifact / intent hygiene (AIDLC)

- Contribution file present with required sections (this file).
- Stage artifacts cite REQUIREMENTS IDs for crypto claims.

### S1 — Install & CLI honesty (FR-CLI-01, FR-DEP-01)

```bash
# Clean editable install from repo root (uv or pip)
pip install -e ".[dev]"
# or: uv pip install -e ".[dev]"

python -c "from fedzk.cli import app"
fedzk --help
fedzk doctor
```

`fedzk doctor` must succeed on a fresh clone (PLAN Phase 0 exit) and, once wired, check artifact path/hash presence (FR-ART-01).

### S2 — Lint / types on touched Python (QUALITY CI target)

```bash
ruff check src/fedzk tests
# if ruff not yet adopted, interim:
# flake8 src/fedzk && black --check src/fedzk
mypy src/fedzk --ignore-missing-imports
```

### S3 — Unit + walking-path tests

```bash
pytest tests/unit -q --tb=short \
  -k "quantiz or prover or verifier or coordinator or cli" \
  --maxfail=1

# Prefer dedicated markers once added:
# pytest -m "bolt1 or walking_skeleton" -q --tb=short
```

### S4 — Golden ZK (crypto gate)

```bash
# Once goldens land under tests/goldens/ (see Golden vector policy)
pytest tests/goldens -q --tb=short
# or package script when added, e.g.:
# python -m fedzk.zk.verify_goldens --circuit-id N_dev
```

Must verify real Groth16 (or documented backend) proofs against frozen public inputs — **not** mocked verify returns.

### S5 — Artifact integrity (FR-ART-01; AIDLC custom sensor)

```bash
# Manifest sha256 matches on-disk circuit/zkey/vkey (paths resolved by doctor / config)
python -c "from pathlib import Path; assert Path('tests/goldens').exists() or Path('src/fedzk').exists()"
# Prefer explicit check once SHA256SUMS exists:
# sha256sum -c tests/goldens/<circuit_id>/SHA256SUMS
```

### S6 — Import / API honesty (FR-API-01)

```bash
python -c "from fedzk.prover.zkgenerator import ZKProver"
python -c "from fedzk.prover.verifier import ZKVerifier"
python -c "from fedzk.zk.input_normalization import GradientQuantizer"
# Reject fictional symbols in test tree:
! rg -n "CoordinatorAPI|ZKGenerator" tests --glob '*.py'
```

### S7 — Security hygiene (SEC-05, NFR-S-01)

```bash
gitleaks detect --source . --no-git -v
bandit -r src/fedzk -lll
# Prod key policy smoke (SEC-01) when mode flag exists:
# FEDZK_ENV=production python -c "..."  # must refuse testkey / empty keys
```

### S8 — Quality-gates runner (reconcile fantasy thresholds)

```bash
python scripts/ci/quality_gates.py --phase bolt1
# Until --phase exists: run the subset that maps to S1–S7 only;
# do not fail Bolt 1 on aspirational prove-latency SLOs.
```

**Sensor matrix summary**

| ID | Must pass for Bolt 1 gate | Blocking |
| --- | --- | --- |
| S1 Install + `fedzk doctor` | Yes | Yes |
| S2 Lint/mypy on touched paths | Yes | Yes |
| S3 Unit / skeleton tests | Yes | Yes |
| S4 Goldens verify | Yes (N_dev OK) | Yes |
| S5 Artifact hashes | Yes when manifest present | Yes |
| S6 Import honesty + no fictional APIs | Yes | Yes |
| S7 gitleaks + bandit high | Yes | Yes |
| Full compliance / e2e enterprise | No | Deferred |

---

## Golden vector policy

Goldens are the cryptographic source of truth for Bolt 1 and later phases (QUALITY §1 Crypto; FR-CIR-02 in Phase 1).

1. **Layout (target):** `tests/goldens/<circuit_id>/` containing at minimum:
   - `input.json` (quantized public/private witness inputs)
   - `proof.json` (snarkjs-shaped Groth16 / `fedzk.proof.v1`)
   - `public.json` (public signals)
   - `SHA256SUMS` (or repo-level manifest) for artifacts + golden files
2. **Generation:** Produce from the real prove path after quantize; record `circuit_id`, `N`, scale/quantize params, backend (`snarkjs` default for Bolt 1).
3. **Verification:** CI re-verifies every golden; failure is a release/bolt blocker. Changing circuit params **requires** regenerating goldens in the same PR.
4. **No mocks:** Default prover must not short-circuit to “always valid”. Test doubles allowed only behind explicit fixtures marked `@pytest.mark.crypto_mock` and excluded from default CI job.
5. **N_dev:** Bolt 1 may freeze goldens at `N≥4` (even if product target is N≥64 in Phase 1). Document `N_dev` in golden metadata; do not claim secure-parameter circuits until FR-CIR-01.
6. **Fail closed:** Missing artifact, hash mismatch, or verify≠1 → sensor failure.
7. **Cross-backend:** Rust parity on goldens is Phase 3 (FR-RST-01); Bolt 1 does not require it.
8. **CONCEPTS sync:** If public statement / constraint meaning changes, update goldens **and** CONCEPTS in the same change set (`construction.md`).

---

## Definition of Done for Bolt 1 walking skeleton

Bolt 1 is done when Phase 0 exit criteria are met for a **real** crypto path, not a demo façade.

**Functional DoD**

- [ ] Fresh clone: install works; `fedzk` console script resolves Typer app (FR-CLI-01).
- [ ] `fedzk doctor` OK; MPC/coordinator default artifact paths resolve to shipped files (FR-ART-01).
- [ ] `GradientQuantizer` (or equivalent) is on the prove path; floats are quantized; non-integers rejected when required (FR-Q-01).
- [ ] End-to-end walking skeleton: **quantize → prove → verify → FedAvg** on `N≥4` with **no mock proofs**.
- [ ] Documented public imports match exports used by the skeleton (FR-API-01).
- [ ] Dual/fictional coordinator APIs removed or quarantined from default tests; e2e uses real APIs.

**Quality / security DoD**

- [ ] Sensors S1–S7 pass on CI (or documented local runner until CI matrix exists).
- [ ] At least one golden vector set verifies under S4.
- [ ] `secrets/` (and similar) untracked; gitleaks clean; prod mode refuses default `testkey` when that flag exists (SEC-01/05).
- [ ] False “production-ready / 100% tested” claims stripped (QUALITY §6).
- [ ] Critical-path coverage trend toward NFR-C-01 on prover/coordinator/quantizer (report in PR; hard fail optional until baseline measured).

**Process DoD**

- [ ] PR cites REQUIREMENTS IDs touched.
- [ ] Conductor gate logged; swarm/parallel units only **after** Bolt 1 approval (`LOOPS.html` / `construction.md`).
- [ ] `quality-gates/config.yaml` items that cannot yet be met are labeled aspirational — not silently failing humans.

**Explicitly out of Bolt 1 DoD:** N≥64 secure circuits, chunk protocol, Rust engine, Flower, Helm, compliance PDF generators, paper attack suite (Phase 1+).

---

## Diary

### Interpretations

- “Comprehensive for crypto” means goldens + fail-closed verify are first-class sensors, not optional extras after unit tests.
- Phase 0 “hygiene” is quality work: installability, honest APIs, secrets, and a real prove/verify path — not enterprise compliance coverage.
- `quality-gates/config.yaml` is intent, not ground truth; Bolt 1 enforces QUALITY.html runnable gates and measured SLOs later.

### Deviations

- Full QUALITY CI pipeline (`lint → unit → golden-zk → integration → rust-cross-verify → docs-smoke → security → wheel`) is the **target**; Bolt 1 may omit rust-cross-verify, compose integration, and SBOM until later phases.
- Coverage ≥90% (NFR-C-01) is a **target** during Bolt 1 if brownfield starts far below; must be measured and ratcheted, not claimed.
- `ruff` appears in QUALITY.html; repo today still lists flake8/black — sensor S2 accepts either until toolchain unified.

### Tradeoffs

- Allowing `N_dev=4` goldens unblocks the walking skeleton without waiting on FR-CIR-01 parameterization — at the cost of not yet proving secure-width circuits.
- Deferring compliance tests avoids greenwashing and keeps CI signal on crypto honesty.
- Strict no-mock default path slows local DX without Node/snarkjs; mitigated by caching golden artifacts in CI (QUALITY §2).

### Open questions

- Exact package path / CLI verb for golden verification (`pytest tests/goldens` vs `fedzk doctor --goldens` vs `scripts/ci/...`)?
- When does artifact `SHA256SUMS` land relative to regenerating vs freezing circom↔artifact sync (PLAN Phase 0 checklist)?
- Is bandit run repo-wide or only on `src/fedzk/{prover,zk,coordinator,mpc}` for Bolt 1 to reduce noise?
- Baseline measured prove latency for N_dev — what becomes `enforced:` vs `aspirational:` in `quality-gates/config.yaml`?
- Marker naming: `@pytest.mark.bolt1` / `walking_skeleton` / `golden` — confirm before CI matrix freeze.
