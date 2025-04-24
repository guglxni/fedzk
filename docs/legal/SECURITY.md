# Security Policy

## Reporting a Vulnerability

We take the security of FedZK seriously. If you believe you've found a security vulnerability in FedZK, please follow these steps to report it to us:

1. **Do not disclose the vulnerability publicly** until it has been addressed by the maintainers.
2. Email your findings to security@fedzkproject.org. If you don't receive a response within 48 hours, please follow up via GitHub.
3. Provide detailed information about the vulnerability, including:
   - Description of the issue
   - Steps to reproduce
   - Potential impact
   - Any suggestions for remediation

## Security Measures

FedZK incorporates several security features:

- **End-to-End Encryption**: All communication between nodes is encrypted
- **Zero-Knowledge Proofs**: Ensures model update integrity without revealing sensitive data
- **Differential Privacy**: Optional noise addition to prevent inference attacks
- **Secure Aggregation**: MPC-based techniques to protect individual updates
- **Input Validation**: Extensive validation to prevent injection attacks

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Security Process

Upon receiving a security report, we will:

1. Confirm receipt of your report within 48 hours
2. Provide an initial assessment of the report within 7 days
3. Work on addressing confirmed vulnerabilities with high priority
4. Keep you informed about our progress
5. Credit you when we publish the fix (unless you prefer to remain anonymous)

## Security Updates

Security updates will be released as soon as possible after a vulnerability is confirmed. We encourage all users to keep their installations up to date.

For additional security information, please refer to the [Implementation Details](../implementation_details.md) document.

## Acknowledgments

We'd like to thank all security researchers who have helped improve the security of FedZK. 