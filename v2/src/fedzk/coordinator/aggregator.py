# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
# Copyright (c) 2025 Aaryan Guglani and FEDzk Contributors
# Licensed under FSL-1.1-Apache-2.0. See LICENSE for details.

"""
DEPRECATED — Dual coordinator collapse (Phase 0 / U11).

Do not add new features here. The canonical HTTP coordinator is:

    uvicorn fedzk.coordinator.api:app

This module re-exports that app so legacy imports
(`from fedzk.coordinator.aggregator import app`) keep working during the
migration window. Tests should target `fedzk.coordinator.api`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "fedzk.coordinator.aggregator is deprecated; use fedzk.coordinator.api:app",
    DeprecationWarning,
    stacklevel=2,
)

from fedzk.coordinator.api import app  # noqa: E402,F401

__all__ = ["app"]
