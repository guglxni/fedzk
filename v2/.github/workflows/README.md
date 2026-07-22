# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""
GitHub Actions only auto-discovers `.github/workflows` at the *repository root*.

Isolation rule: do not modify the upstream root tree for v2 work.
This file under `v2/.github/workflows/` is the **canonical workflow source** to copy
into a thin root pointer later (human-proxy / separate PR that only adds a one-file
include). Until then, use:

    v2/scripts/ci_local.sh
"""
