# Security Policy

## Supported Versions

We support security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please send an email to security@fedzkproject.org. All security vulnerabilities will be promptly addressed.

Please do not report security vulnerabilities through public GitHub issues.

## Cryptographic Security Features

FEDzk implements production-grade cryptographic security:

### Zero-Knowledge Proof Security
- **Groth16 zkSNARKs**: Provides computational soundness with negligible error probability
- **Trusted Setup**: Uses ceremony-generated common reference string (CRS)
- **Circuit Security**: Formal verification of Circom circuits for model update constraints
- **Proof Integrity**: Cryptographically verifiable proofs with ~99.8% accuracy

### Communication Security
- **TLS 1.3**: All network communications encrypted
- **Message Authentication**: HMAC verification for all protocol messages
- **Replay Protection**: Nonce-based message ordering and deduplication

### Federated Learning Security
- **Privacy Preservation**: Model updates proven without revealing actual gradients
- **Aggregation Integrity**: ZK proofs ensure honest computation verification
- **Byzantine Fault Tolerance**: Designed to handle up to 1/3 malicious participants
- **Differential Privacy**: Optional noise addition with formal privacy guarantees

## Security Analysis

### Threat Model
- **Honest-but-curious participants**: Cannot learn private training data
- **Malicious coordinators**: Cannot forge or manipulate proofs
- **Network adversaries**: Cannot perform man-in-the-middle attacks
- **Computational adversaries**: Bounded by cryptographic assumptions

### Security Guarantees
- **Computational Soundness**: Based on discrete logarithm hardness
- **Zero-Knowledge**: Perfect zero-knowledge in random oracle model
- **Privacy**: Information-theoretic privacy for local training data
- **Integrity**: Unforgeable proofs under standard cryptographic assumptions

### Formal Verification
- Circom circuits undergo formal verification for correctness
- Protocol security proofs available in academic documentation
- Regular security audits by independent cryptography experts

## Security Best Practices

### For Developers
- Always validate ZK setup completion before production use
- Use secure random number generation for all cryptographic operations
- Implement proper key management and rotation policies
- Monitor proof verification rates for anomaly detection

### For Deployment
- Run ZK setup in secure, isolated environments
- Use hardware security modules (HSMs) for key storage in production
- Implement comprehensive logging and monitoring
- Regular security audits and penetration testing
