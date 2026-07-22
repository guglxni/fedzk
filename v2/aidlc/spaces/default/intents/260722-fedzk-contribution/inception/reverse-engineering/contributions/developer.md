# Contribution: aidlc-developer-agent

Intent: `260722-fedzk-contribution`  
Stage: inception / reverse-engineering  
Focus: brownfield code truth for CLI, quantize→prove, MPC artifacts, coordinator dual stack, e2e imports, tracked secrets  
Method: shell/read + `graphify query` (no nested agents)

## Code truth vs README claims

| README / packaging claim | Code truth (evidence) |
|---|---|
| Console entry `fedzk = "fedzk.cli:app"` (`pyproject.toml:98-99`, `setup.py:161`) | **Broken.** `importlib` resolves `fedzk.cli` to empty package `src/fedzk/cli/__init__.py` (21 bytes, no exports), **not** `src/fedzk/cli.py` (494 lines, real Typer `app`). Spec origin: `.../src/fedzk/cli/__init__.py`; `hasattr(cli, "app") == False`. |
| `fedzk server start` (README checklist L115) | **No such command.** Real MPC command is `@mpc_app.command("serve")` → `fedzk mpc serve` in `src/fedzk/cli.py` (unreachable via entrypoint until shadowing fixed). |
| `setup.py` also ships `fedzk-admin`, `fedzk-coordinator`, `fedzk-worker` → `admin_app` / `coordinator_app` / `worker_app` | **Phantom.** Those names are not assigned in `src/fedzk/cli.py` (only `app`, `client_app`, `mpc_app`, `benchmark_app`, `main`). |
| `fedzk setup` wraps ZK setup | Script path is `Path(__file__).parent / "scripts" / "setup_zk.sh"` → would be `src/fedzk/scripts/setup_zk.sh` (missing). Real script: `scripts/setup/setup_zk.sh`. |
| “Integer Gradient Processing” / “gradients must be quantized” (README L92, L101–102) | **`ZKProver` never calls quantize.** Zero matches for `quantize` / `GradientQuantizer` / `input_normalization` in `src/fedzk/prover/zkgenerator.py`. `_prepare_input_standard` / `_prepare_input_secure` flatten via `.tolist()` and pass floats through as `"gradients"`. Quantizers live in `src/fedzk/zk/input_normalization.py` and circuit helper `model_update_quantized.py`; wired from `advanced_verification.py` / demos / unit component tests—not the prove path. |
| “Exactly 4 gradient values per proof” | **True for circuit shape** (`max_inputs=4` truncate/pad), but README presents it as a polished product constraint while silent truncation of longer updates remains a correctness footgun. |
| Dual “production coordinator” story | **Two FastAPI stacks:** `coordinator/aggregator.py` (simple verify + in-memory FedAvg) vs `coordinator/api.py` → `logic.py` (richer validation). `api.py` does **not** import aggregator. Benchmarks use `logic`; several unit tests still import aggregator (and invent `SecureAggregator`). |
| “100% Tested” / “Production Ready” / “Zero mock” / “Enterprise Deployed” (README badges L25, L40, L952) | **Marketing > tree.** E2E module cannot import its own public surface (see below); CLI entrypoint is dead; MPC default WASM paths miss on disk; **234** secret-like paths tracked in git. |
| Top-level `cli/` package (scan focus) | **No repo-root `cli/`.** Only `src/fedzk/cli.py` + empty `src/fedzk/cli/` package—shadowing is the brownfield bug. |

## Broken paths (evidence with file paths)

### 1. CLI: module vs package shadowing
- **Real CLI:** `src/fedzk/cli.py` — Typer apps, commands `setup` / `generate` / `verify`, `mpc serve`, `benchmark run`, `client train|prove`.
- **Shadow package:** `src/fedzk/cli/` with `__init__.py` only → wins on `import fedzk.cli`.
- **Entrypoints:** `pyproject.toml` `[project.scripts] fedzk = "fedzk.cli:app"`; `setup.py` console_scripts same + three missing apps.
- **Redirect:** `src/fedzk/__main__.py` does `from fedzk.cli import main` — also fails under package shadow (`main` not on empty package).

### 2. Quantize unwired from `zkgenerator.py`
- Prove entry: `ZKProver.generate_proof` → `_prepare_input_*` → snarkjs (`src/fedzk/prover/zkgenerator.py` L126–246).
- Asset paths for prover correctly use `ASSET_DIR = .../src/fedzk/zk/` (`model_update.wasm`, `proving_key.zkey`, …) — those files **exist**.
- Quantize API exists (`GradientQuantizer.quantize_gradients` in `src/fedzk/zk/input_normalization.py`) but is **not imported** by prover.
- Circuit-side quantize helper: `src/fedzk/zk/circuits/model_update_quantized.py` (`ModelUpdateQuantizedCircuit.quantize_gradients`) also unused by `ZKProver`.
- Demo/docs often `from fedzk.utils import GradientQuantizer` — fragile relative to package layout; canonical module is `fedzk.zk.input_normalization`.

### 3. MPC server artifact path mismatch
File: `src/fedzk/mpc/server.py` L63–77.

| Default env | Resolved default | Exists? |
|---|---|---|
| `MPC_STD_WASM_PATH` | `{PROJ_ROOT}/src/fedzk/zk/circuits/build/model_update.wasm` | **No** (`circuits/build/` absent) |
| `MPC_SEC_WASM_PATH` | `.../circuits/build/model_update_secure.wasm` | **No** |
| `MPC_STD_ZKEY_PATH` | `.../circuits/proving_key.zkey` | Yes (symlink under circuits) |
| `MPC_SEC_ZKEY_PATH` | `.../circuits/proving_key_secure.zkey` | Yes |
| `MPC_STD_VER_KEY_PATH` / secure | `.../circuits/verification_key*.json` | Yes (symlinks) |

**Contrast:** lifespan validation uses `Path(__file__).parent.parent / "zk"` (= `src/fedzk/zk`), where WASM **does** exist—but prove/verify handlers still default to the broken `circuits/build/...` WASM paths unless env overrides. Health check L545–551 will warn on missing circuit files using those defaults.

`PROJ_ROOT = parent×4` from `src/fedzk/mpc/server.py` is correct for repo root; the **subdirectory** `circuits/build/` is the lie.

### 4. Coordinator: aggregator vs api/logic
- `src/fedzk/coordinator/aggregator.py` (~99 LOC): own FastAPI `app`, `UpdateSubmission`, `submit_update`, `get_status`, verifier on `zk/verification_key.json`.
- `src/fedzk/coordinator/api.py` (~166 LOC): FastAPI `app` (not a `CoordinatorAPI` class), delegates to `logic.submit_update` / `get_status` / security stats.
- `src/fedzk/coordinator/logic.py` (~786 LOC): `VerifiedUpdate`, `AggregationBatch`, imports `fedzk.prover.advanced_proof_validator` (file **exists** under `src/fedzk/prover/`).
- **Orphan / dual API:** production narrative is unclear which app to run; tests disagree (`test_integration.py` / `test_aggregator.py` → aggregator; `benchmark/end_to_end.py` → logic/api).
- **Fictional symbol:** `tests/unit/test_component_testing.py` imports `SecureAggregator` from aggregator — **no such class in `src/`**.

### 5. E2E fictional imports
File: `tests/e2e/test_full_workflow.py` L15–18:

```text
from fedzk.coordinator.api import CoordinatorAPI      # no class; module exports FastAPI `app`
from fedzk.client.trainer import FederatedTrainer   # actual class: LocalTrainer
from fedzk.prover.zkgenerator import ZKGenerator    # actual class: ZKProver
from fedzk.validation.proof_validator import ProofValidator  # exists; ctor wants ProofValidationConfig, not free-form dict “config=”
```

Fixture then constructs `CoordinatorAPI(config=...)`, `FederatedTrainer(config=...)`, `ZKGenerator(config=...)` — none of those call signatures exist on real types. Module fails at collection/import before any workflow assertion runs. Patches target nonexistent methods (`start_training_round`, `train_local_model`).

### 6. Secrets tracked count
Command (repo root):

```bash
git ls-files | rg -i 'secret|\.pem$|\.key$|\.env($|\.)|credential|password|\.p12|\.pfx|id_rsa|\.enc$|private' | wc -l
```

**Result: 234** paths.

Breakdown (approx):
- `secrets/` directory alone: **223** tracked `*.enc` files (includes `prod_api_key.enc`, `prod_db_password.enc`, `prod_jwt_secret.enc`, many `concurrent_*` / `perf_*` / `monitor_*` test blobs).
- Plus docs/demos/helm/config/tests naming “secret”, `.env.example`, archives under `archives/backups/secrets/`.
- Working tree also has empty `.env` (untracked / ignore-dependent); still a hygiene smell next to tracked ciphertext corpus.

## Suggested Phase 0 fix order

1. **Unblock CLI (FR-CLI-01):** Delete or repurpose empty `src/fedzk/cli/` package so `fedzk.cli` loads `cli.py`; or move Typer app into `cli/__init__.py` / `cli/main.py` and delete the sibling module. Verify `python -c "from fedzk.cli import app"` and `fedzk --help`. Align `setup.py` entrypoints (drop phantom admin/coordinator/worker or implement them). Fix `setup` script path → `scripts/setup/setup_zk.sh`. Update README `fedzk server start` → `fedzk mpc serve`.

2. **Unify ZK artifact roots (FR-ART-01):** Point `mpc/server.py` defaults at `src/fedzk/zk/{model_update.wasm,proving_key.zkey,verification_key.json}` (same as `ZKProver.ASSET_DIR`), not `zk/circuits/build/`. Add a single path helper + env overrides; make `/health` fail loud if WASM missing.

3. **Wire quantize on prove (FR-Q-01):** In `ZKProver._prepare_input_*` (or a pre-step in `generate_proof`), call `GradientQuantizer` (or adaptive), emit integers only, attach scale metadata to public signals / API. Reject floats when quantization disabled. Stop documenting demos that import a non-canonical `fedzk.utils.GradientQuantizer` without fixing that export.

4. **Collapse coordinator dual stack (FR-CRD-01):** Choose `api.py` + `logic.py` as canonical; demote `aggregator.py` to thin re-export or delete after migrating tests. Remove `SecureAggregator` fiction from component tests.

5. **Rewrite or quarantine e2e (FR-API-01):** Replace fictional imports with `LocalTrainer`, `ZKProver`, FastAPI `app` / TestClient against `api.py`, real `ProofValidator` config type—or mark e2e skipped until skeleton runs. Do not claim “100% tested” while collection fails.

6. **Secrets hygiene (SEC-05 / NFR-S):** Stop tracking `secrets/*.enc` (especially `prod_*`); gitignore + rotate anything that was ever real; keep only documented fixtures under `tests/fixtures/` if needed. Re-count with the same `git ls-files | rg ...` gate in CI (expect ~0 ciphertext blobs).

7. **Docs honesty pass:** Downgrade README production/100%/enterprise badges until Phase 0 gates pass; document n=4 + chunk roadmap explicitly.

## Diary

### Interpretations
- Brownfield debt is concentrated at **boundaries** (CLI packaging, artifact paths, public type names), not absence of ZK code—the prover/WASM under `src/fedzk/zk/` look more real than README/e2e claim surface.
- “cli/ vs cli.py” in the scan brief maps to **`src/fedzk/cli/` vs `src/fedzk/cli.py`**; there is no top-level `cli/`.
- Quantize is a **library island**: implemented and lightly tested, disconnected from the snarkjs prove pipeline the product markets.
- Dual coordinators read as unfinished migration (simple aggregator → logic) with tests split across eras.

### Deviations
- Did not run full pytest (deps like FastAPI missing in bare `PYTHONPATH=src` probe); relied on static path existence, `importlib` resolution, and symbol search.
- Secrets “count” uses a deliberate regex over `git ls-files` (234)—includes docs/helm naming “secret”, not only decryptable material; still the right Phase 0 hygiene metric for this intent.
- `advanced_proof_validator.py` **is present** in `src/`; earlier graph noise and missing runtime deps should not be mistaken for a missing module.

### Tradeoffs
- Prefer **deleting the empty `cli/` package** over moving 494 lines—smallest diff to restore entrypoint—unless the intent is a multi-module CLI package (then move code into `cli/` and delete `cli.py`).
- Prefer **one artifact root** (`src/fedzk/zk/`) over resurrecting `circuits/build/`; circuits/ already holds sources + intermediate zkeys.
- Keeping aggregator as a compatibility shim risks prolonging dual APIs; Phase 0 should pick a winner.

### Open questions
- Were any `secrets/prod_*.enc` ever production credentials, or only test names? Assume hostile until rotated.
- Is remote MPC prove (`FR-MPC-01`) in Phase 0 or Phase 2? Artifact fix is Phase 0 either way.
- Should `model_update_quantized` circuit become the default prove circuit, or stay optional while standard circuit receives integerized inputs only?
- Which coordinator port/process do Docker/Helm actually launch today—aggregator or api?
