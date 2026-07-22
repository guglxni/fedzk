# Contribution: aidlc-product-agent

Intent: `260722-fedzk-contribution`  
Stage: inception / requirements-analysis (+ ideation intent affirmation)  
Scope path: **brownfield → mvp → paper+OSS**  
Normative kit: CONCEPTS, PROPOSAL, REQUIREMENTS, `aidlc/spaces/default/memory/project.md`

## Positions

1. **Product one-liner (locked):** Only updates that can prove they obey the integrity policy get into the global model—client-side Groth16 proofs of *update constraint satisfaction*, not “fully private FL.”
2. **Contribution thesis (affirm PROPOSAL):** Ship an open, end-to-end framework + empirical study for client-side zkSNARK proofs of federated model-update integrity, complementary to aggregator-focused zkFL. Primary novelty for paper: *runnable open artifact pipeline* + *client-update ZK* + *honest chunked capacity (N)*; systems (Rust prove path) is secondary.
3. **Journey order is non-negotiable:**
   - **Brownfield first:** make the existing tree installable and truthful (CLI, deps, artifact paths, documented imports, secrets hygiene)—no paper claims until the walking skeleton runs.
   - **MVP next:** parameterized secure circuit (N≥64), quantize-on-prove, chunk protocol, single coordinator with verify-before-FedAvg.
   - **Paper+OSS last gate:** schema-valid experiment transcripts, attack rejection eval, accuracy vs plain FL, limitations that match reality (CONCEPTS §15 / RS-04).
4. **MVP scope = AIDLC `mvp` (Phase 0–2 walking skeleton):** comprehensive depth for prover/coordinator; Flower wrapper (FR-FLW-01) and “enterprise ops” stay out of the first shippable contribution.
5. **Honesty over marketing:** docs/paper must say development zkeys are not production-trusted; ZK ≠ beneficial model / clean data; quantization and chunking can change learning dynamics—report deltas.
6. **Brownfield debt is product risk:** CLI shadowing, hard torch import, n=4 circuits, unwired quantizer, wrong MPC artifact defaults, dual coordinators, fictional e2e APIs, and tracked secrets block “contribution-grade” until cleared or explicitly demoted in docs.

## Requirements priorities (top 10 with FR- IDs from kit)

Ordered for brownfield → mvp → paper+OSS (not raw table order). All IDs from `docs/agent-kit/REQUIREMENTS.html`.

| Rank | ID | Why this rank for this intent |
|------|-----|-------------------------------|
| 1 | **FR-CLI-01** | Brownfield unblock: `fedzk` must resolve the Typer app so demos/CI/docs are real. |
| 2 | **FR-DEP-01** | Brownfield unblock: install path must include or clearly extra-doc torch so examples do not lie. |
| 3 | **FR-ART-01** | Brownfield unblock: default MPC/coordinator artifact paths must hit shipped files (no silent broken prove). |
| 4 | **FR-API-01** | OSS trust: documented imports match exports; kill fictional e2e surface area in public docs. |
| 5 | **FR-Q-01** | Product core (CONCEPTS §9): prove path quantizes floats; reject non-integers otherwise. |
| 6 | **FR-CIR-01** | MVP crypto: parameterized N≥64 secure circuit + regenerated artifacts (leave n=4 behind). |
| 7 | **FR-CIR-02** | OSS/CI trust: artifact manifest with sha256; CI verifies what paper cites. |
| 8 | **FR-CHK-01** | Honest capacity (P4): chunk protocol for updates longer than N—no silent truncate in API or paper. |
| 9 | **FR-CRD-01** | Product core: single coordinator; verify-before-aggregate FedAvg (collapse dual coordinators). |
| 10 | **FR-EXP-01** | Paper+OSS gate: round-runner produces schema-valid transcripts others can cite (RS-01–03). |

**Immediately after top 10 (still P0 / contribution-critical, not dropped):** **FR-MPC-01** (Phase 2 remote prove with auth; no default prod keys / SEC-01)—required for mvp completeness, ranked after the paper transcript path because contribution value for venues hinges on reproducible experiments first. **FR-BAT-01** (P1) and **FR-RST-01** (P1) support paper latency tables / optional Rust engine; **FR-FLW-01** (P2) is post-mvp integration sugar.

**NFRs that police the same journey:** NFR-R-01 (reproducible transcript hashes), NFR-D-01 (docs examples in CI), NFR-S-01 + SEC-05 (no critical vulns; untrack secrets), NFR-P-01/P-02 (report prove/verify latency honestly), NFR-C-01 (≥90% on prover/coordinator/quantizer).

## Non-goals to police in docs/paper

Agents and humans must reject or rewrite wording that claims the following for v1 (CONCEPTS §3, §14–15; PROPOSAL §5):

1. **Hiding the update from the coordinator** — FedAvg still consumes ĝ; do not equate FEDzk with secure aggregation.
2. **“I trained correctly on my private dataset”** — v1 proves properties of the *submitted update*, not full training traces.
3. **Aggregator honesty / “FedAvg used all clients fairly”** — that is zkFL-style; cite as complementary future work, never as FEDzk v1.
4. **On-chain settlement as required** — optional; not needed for science or OSS usefulness.
5. **“Fully private federated learning” / “removes all FL leakage”** — unless DP + secure aggregation are in scope *and* evaluated; prefer “ZK proofs of update constraint satisfaction.”
6. **Malicious aggregator rewriting averages** — out of threat model v1; do not claim defense.
7. **In-bound adversarial updates impossible** — updates inside L can still poison; say so in limitations.
8. **Silent capacity miracles** — never imply arbitrary model dims without chunking; N and chunk protocol must be visible in API/paper (P4).
9. **Dev zkeys as production trust** — ceremony / Powers of Tau story required; toxic waste called out (SEC-03).
10. **Legacy code as the concept** — n=4 / unwired quantize do *not* instantiate CONCEPTS yet; do not paper results from broken paths.

## Diary

### Interpretations

- Ideation intent **affirmed**: build FEDzk into a contribution-grade OSS ZK-FL framework and research platform (client-side Groth16 update integrity + reproducible experiments), with primary AIDLC scope `mvp` then promote for ops.
- “Contribution-grade” means: installable, single coherent coordinator story, circuits/artifacts that match claims, and citeable transcripts—not a larger feature checklist.
- Product success for this intent is dual: (a) a stranger can run train → quantize → prove → verify → aggregate; (b) a reviewer can reproduce accuracy/attack tables from open transcripts.

### Deviations

- Ranked **FR-EXP-01** inside the top 10 ahead of **FR-MPC-01** even though both are P0 in the kit, because paper+OSS citation value depends on the experiment harness; remote prove remains Phase-2 mvp work, not demoted from the catalog.
- Deferred **FR-FLW-01** and enterprise/ops packaging beyond Phase 0–2 despite ecosystem pressure—Flower is an experiment backend later, not the product core.
- Treat fictional e2e APIs and dual coordinators as **doc/product defects** to remove under FR-API-01 / FR-CRD-01, not as alternate supported products.

### Tradeoffs

- **Integrity vs privacy marketing:** tighter truthful positioning reduces hype but increases publishability and OSS trust.
- **N≥64 + chunking vs toy n=4 demos:** slower path to green CI, required for honest capacity and paper realism.
- **Single coordinator now vs preserving both code paths:** short-term churn; long-term clearer threat model and FR-CRD-01 compliance.
- **Paper-ready transcripts before polish UX:** favors research/OSS citation over feature breadth (batch verify, Flower) in the first contribution cut.
- **Quantization fidelity:** field-safe integers enable Groth16; must budget accuracy-delta reporting (RS-01) instead of hiding the cost.

### Open questions

1. Primary paper venue bias (systems vs crypto-adjacent ML): does that change weight of FR-RST-01 / NFR-P-01 vs attack-suite depth (RS-02)?
2. Minimum model/dataset for “contribution” MVP demos vs paper tables (MNIST walking skeleton vs CIFAR config in FR-BAT-01 / prd)—what is the first public claim set?
3. Circuit policy L for v1 secure circuit: exact public bounds (L2, min nonzero, round binding fields)—freeze before FR-CIR-01 regen?
4. Chunk commitment / binding scheme: product-visible schema in proof envelope—who owns the normative wire format relative to `fedzk.proof.v1`?
5. When is milestone `proposal-accepted` / STACK lock considered done for agents (human tag vs conductor human-proxy-approve on docs only)?
6. Secrets purge (223 tracked): is SEC-05 a hard gate before any public paper preprint pointer lands in README?
