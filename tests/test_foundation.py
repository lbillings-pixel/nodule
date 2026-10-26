"""The package installs, and agatesim is importable as its library (ADR 0002)."""

import agatesim

import nodule


def test_nodule_has_a_version() -> None:
    assert nodule.__version__


def test_agatesim_is_available_as_a_library() -> None:
    from agatesim.models.base import Domain

    assert agatesim.__version__
    assert Domain is not None
