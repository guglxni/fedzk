# Functional Source License 1.1 with Apache-2.0 Future Grant (FSL-1.1-Apache-2.0)
"""Flower adapter unit tests (no flwr required)."""

from __future__ import annotations

import torch

from fedzk.adapters.flower import (
    FEDzkClientMixin,
    ndarray_list_to_named,
    try_import_flwr,
)


def test_ndarray_list_to_named():
    named = ndarray_list_to_named([[0.1, 0.2], [0.3]], names=["w", "b"])
    assert list(named.keys()) == ["w", "b"]
    assert named["w"].shape == (2,)


def test_mixin_prove_smoke():
    m = FEDzkClientMixin(n_circuit=4)
    grads = {"w": torch.randn(4) * 0.001}
    proved = m.prove_numpy_grads(grads)
    assert proved["mode"] == "single"
    metrics = m.pack_fit_metrics(proved)
    assert metrics["fedzk_mode"] == "single"


def test_try_import_flwr_tuple():
    ok, mod = try_import_flwr()
    assert isinstance(ok, bool)
    if ok:
        assert mod is not None
