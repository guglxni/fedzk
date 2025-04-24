# ZK Circuit Components

This directory contains the zero-knowledge proof circuit components used by FedZK.

## Overview

FedZK uses zero-knowledge proofs to verify the integrity of gradient updates without revealing the actual gradient values. This directory contains the compiled artifacts of the Circom circuits used for proof generation and verification.

## Files

- `model_update.wasm`: WebAssembly file for the basic model update circuit
- `proving_key.zkey`: Proving key for the basic model update circuit
- `verification_key.json`: Verification key for the basic model update circuit
- `model_update_secure.wasm`: WebAssembly file for the secure model update circuit (with constraints)
- `proving_key_secure.zkey`: Proving key for the secure model update circuit
- `verification_key_secure.json`: Verification key for the secure model update circuit

## Usage

These files are automatically used by the `ZKGenerator` and `ZKVerifier` classes. You don't need to interact with them directly unless you're customizing the circuit implementation.

## Regenerating Artifacts

To regenerate these files (e.g., after modifying the circuits), run:

```bash
python -m fedzk.cli setup --force
```

## Third-Party Components

The circuit implementation in the `/circuits` directory includes components from the Circom standard library, which are licensed under the Apache 2.0 License. See [THIRD_PARTY_LICENSES.md](../circuits/THIRD_PARTY_LICENSES.md) for details.

## License

These circuit artifacts are generated from circuits covered by the MIT License, as specified in the main project LICENSE file.

Last updated: April 24, 2025 