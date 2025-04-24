# Security Policy

## Supported Versions

FedZK is currently under active development. We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

The FedZK team takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose your findings and will make every effort to acknowledge your contributions.

### How to Report a Vulnerability

Please report security vulnerabilities by emailing **security@fedzk.org**. We strongly recommend encrypting your report using our PGP key (see below) and signing your message with your own GPG key if possible.

**PGP Key for Encrypted Reports:**

```
-----BEGIN PGP PUBLIC KEY BLOCK-----
mDMEZfEK1hYJKwYBBAHaRw8BAQdA9EWz7VdXKL+QJ1v94RVg2QkpOcjH9UTk3dE1
jL3zK8+0JkZlZFpLIFNlY3VyaXR5IFRlYW0gPHNlY3VyaXR5QGZlZHprLm9yZz6I
kAQTFggAOBYhBPy1CjLEgDeJr4uJPtA3VCkivsR5BQJl8QrWAhsDBQsJCAcCBhUK
CQgLAgQWAgMBAh4BAheAAAoJENA3VCkivsR5VWQBAI+DMNfyLg+NVfBVr3VVgQjt
QL3bSjQYR6WDw6+e9ASnAP9z8tIw8fkXvs4YN1vuQHmYOz3dvgRYAL55GcwyPbYF
CbgzBGXxCtYSCisGAQQBl1UBBQEBB0AnnVZCTJmT9JZhwx3zr0gOVj+W7oa59MFy
+YtZ3dGHTgMBCAeIfgQYFggAJhYhBPy1CjLEgDeJr4uJPtA3VCkivsR5BQJl8QrW
AhsMBQkJZgGAAAoJENA3VCkivsR5nwEA/jV+v9RXXL3gBnNc+9B1d0ypEjiQXr1O
jU9Cy3L7y+oDAP9puZvOMTe8zr09im9QuxueXHfQP5kyUnPZsTfPv2GGBA==
=sK74
-----END PGP PUBLIC KEY BLOCK-----
```

Key fingerprint: `FCB5 0A32 C480 3789 AF8B 893E D037 5429 22BE C479`

When reporting a vulnerability, please include:
- A detailed description of the vulnerability
- Steps to reproduce the issue
- Potential impact of the vulnerability
- CVSS v3.1 score if possible (see Severity Classification below)
- Suggestions for addressing the issue (if available)

### Response Timeline

- **Initial Response**: We aim to acknowledge receipt of your vulnerability report within 48 hours.
- **Status Update**: We will provide a more detailed response within 7 days, indicating the next steps in handling your submission.
- **Vulnerability Confirmation**: We will work to confirm the vulnerability and determine its impact.
- **Fix Development**: If applicable, we will develop a fix and coordinate with you on the disclosure timeline.

### Disclosure Policy

We follow a coordinated disclosure process:

1. Once a security vulnerability is confirmed, we will work on a fix.
2. We will release the fix and publicly disclose the vulnerability after the fix is available to users.
3. We will credit the reporter (unless anonymity is requested) in the disclosure.
4. Security advisories will be published in our GitHub repository's Security tab.

## Severity Classification

We use the Common Vulnerability Scoring System (CVSS v3.1) to assess vulnerability severity. Our severity levels map to CVSS scores as follows:

| Severity | CVSS v3.1 Score |
|----------|----------------|
| Critical | 9.0 - 10.0     |
| High     | 7.0 - 8.9      |
| Medium   | 4.0 - 6.9      |
| Low      | 0.1 - 3.9      |

### Critical (CVSS 9.0 - 10.0)

- Remote code execution with high impact
- Exposure of cryptographic keys used in proof generation or verification
- Vulnerabilities allowing bypass of zero-knowledge proof verification
- Access to private training data through side-channel attacks

### High (CVSS 7.0 - 8.9)

- Authentication bypass in MPC server
- Unauthorized access to coordinator or client data
- Denial of service attacks affecting entire training process
- Vulnerabilities in the integrity of the federated learning process

### Medium (CVSS 4.0 - 6.9)

- Information disclosure of non-sensitive data
- Denial of service requiring significant resources
- Cross-site scripting in web interfaces (if applicable)
- Issues affecting performance but not security of proofs

### Low (CVSS 0.1 - 3.9)

- Minor information disclosure
- Vulnerabilities requiring unlikely user interaction
- Limited impact denial of service issues
- Configuration weaknesses

For guidance on calculating CVSS scores, please refer to the [CVSS v3.1 Specification](https://www.first.org/cvss/v3.1/specification-document).

## Security Best Practices

When using FedZK, we recommend the following security best practices:

1. **API Key Management**: Rotate API keys regularly and use unique keys for different deployments.
2. **Network Security**: Deploy the MPC server and coordinator behind appropriate firewalls and use TLS for all communications.
3. **Environment Variables**: Securely manage environment variables containing sensitive configuration.
4. **Regular Updates**: Keep FedZK and its dependencies up to date.
5. **Access Control**: Implement proper access controls for all components.

## Security Updates and Announcements

Security announcements will be made via:
- GitHub Security Advisories on our repository: https://github.com/aaryanguglani/fedzk/security/advisories
- The FedZK mailing list (security-announce@fedzk.org)

All security-related communications from our team will be signed with our GPG key when possible.

Last updated: April 24, 2025 