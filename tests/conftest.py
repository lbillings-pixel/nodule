from pathlib import Path

import pytest

from nodule.specimen import Specimen, load_preset

REPO_ROOT = Path(__file__).resolve().parents[1]

# The first look cut's lens and drift settings (ADR 0016 s.5, set 1); all `effective`.
LOOK = dict(lens_strength=1.5, lens_length_mm=4.0, lens_thickness_mm=0.1, lens_cover=0.2,
            translucency_drift=0.25, translucency_drift_length_mm=20.0,
            translucency_drift_thickness_mm=0.5)  # fmt: skip


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def look() -> Specimen:
    """n2-slice with lenses and drift (~3 s to build)."""
    cfg = load_preset("n2-slice")
    model = cfg.volume.model.model_copy(update=LOOK)
    volume = cfg.volume.model_copy(update={"model": model})
    return Specimen.build(cfg.model_copy(update={"volume": volume}))
