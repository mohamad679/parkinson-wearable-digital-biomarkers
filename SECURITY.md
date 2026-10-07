# Security Policy

## Reporting a Vulnerability

We take security seriously, especially in research software that may eventually be used in health-related contexts.

If you discover a security vulnerability, **please email mohamad679@github.com** instead of opening a public issue. Include:

1. **Description** of the vulnerability
2. **Affected component** (e.g., data loading, model fitting)
3. **Steps to reproduce** (minimal example if possible)
4. **Potential impact** (e.g., data leakage, code execution)
5. **Suggested fix** (if you have one)

We will:
- Acknowledge receipt within 48 hours
- Investigate and confirm the issue
- Work on a fix and release a patched version
- Credit you in release notes (unless you prefer anonymity)

## Security Considerations for Users

This is research-grade software. Users should be aware of:

### Data Protection
- **No encryption by default**: CSV data is processed in plaintext. Implement your own encryption for sensitive participant data.
- **Memory exposure**: Intermediate arrays and model parameters remain in memory during execution.
- **Temporary files**: Scripts may create temporary files in your working directory. Use `tmpdir` isolation for sensitive data.

### Model Security
- **No access control**: The code does not authenticate users or control data access.
- **Reproducibility via seeds**: Models are deterministic; using the same seed produces identical results (which may be a security concern in some contexts).
- **Feature leakage**: Ensure proper subject-aware validation; naive random splitting will leak subject-level information.

### Deployment Safety
- **Research use only**: This is not a clinically validated device. Never deploy without appropriate governance, validation, and approval.
- **No guarantees**: Model predictions are estimates with uncertainty; they should never be the sole basis for clinical decisions.

## Dependency Security

We use:
- **scikit-learn** for models
- **pytest**, **ruff** for development/testing
- Standard library for core utilities

Dependencies are pinned in `requirements.txt`. We recommend:
1. Regularly updating dependencies: `pip install --upgrade -r requirements.txt`
2. Monitoring security advisories via `pip audit`
3. Using a virtual environment to isolate this project

To check for known vulnerabilities:
```bash
pip install pip-audit
pip-audit
```

## Secure Usage Guidelines

### For Participant Data

1. **Minimize collection**: Collect only the accelerometer channels and subject IDs you need.
2. **Secure storage**: Keep raw data encrypted at rest and in transit.
3. **Access control**: Limit who can access raw data and model outputs.
4. **Retention policy**: Plan how long to retain data and model artifacts.
5. **Compliance**: Ensure compliance with ethics, IRB, GDPR, HIPAA, or other relevant regulations.

### For Model Development

1. **Subject boundaries**: Never mix training and test subjects; always use subject-aware splitting.
2. **Validation isolation**: Keep data preprocessing, feature selection, and calibration inside training folds.
3. **Blind evaluation**: Report test-set performance without iterating on test data.
4. **Reproducibility audit**: Include random seeds and configuration in all reports.

## Incident Response

If you suspect a breach or misuse of this software:
1. Contact mohamad679@github.com
2. Include as much context as possible
3. We will work with you to understand and address the issue

## Security Best Practices for Contributors

When contributing code:
- Do not commit secrets, credentials, or tokens
- Do not commit real participant data
- Validate all user inputs (CSV schema is validated, but add checks for new inputs)
- Use parameterized operations to avoid injection vulnerabilities
- Document any security assumptions or limitations in your code comments

## References

- [OWASP Secure Coding Practices](https://owasp.org/www-community/Secure_Coding_Practices)
- [GitHub Security Best Practices](https://docs.github.com/en/code-security)
- [Model Card for Model Reporting](https://arxiv.org/abs/1810.03993)

---

Thank you for helping keep this research software secure and trustworthy.
