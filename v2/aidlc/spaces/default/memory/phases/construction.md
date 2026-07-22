# Construction phase rules — FEDzk

- Walking skeleton Bolt 1: quantize → prove → verify → FedAvg on N≥4 with goldens.
- No mock proofs; fail closed.
- Swarm/parallel units allowed after Bolt 1 approved (docs/CI vs circuits may parallelize).
- Every crypto PR updates goldens + CONCEPTS if statement changes.
- Sensors: circuit artifact hashes, CLI import check, e2e import sanity.
