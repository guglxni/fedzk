# Organization / framework defaults (FEDzk + AIDLC)

Adapted from [awslabs/aidlc-workflows](https://github.com/awslabs/aidlc-workflows/tree/v2) org posture for this Cursor-hosted loop.

## Way of Working

- Trunk-based development; short-lived branches; small PRs.
- Every stage produces artifacts under the active intent record dir.
- Conductor (orchestrator) owns all delegation — agents never invoke each other.
- Approval gates are mandatory at phase boundaries; in Cursor loops the conductor may act as **human-proxy** only when the user has granted loop autonomy (see `project.md`).

## Testing Posture

- No mock proofs on default cryptographic paths (FEDzk CONCEPTS P2).
- Golden vectors for every circuit_id before claiming verify works.
- Critical path coverage ≥90% (prover, coordinator, quantizer).

## Walking Skeleton

- Greenfield/MVP: ship a walking skeleton (quantize → prove → verify → FedAvg) before enterprise scaffolding.
- Prefer Phase 0 hygiene before new features.

## Deployment

- Dev keys labeled untrusted; production refuses default API keys.
- Prefer Rust prove sidecar when Node is undesirable in prod images.
