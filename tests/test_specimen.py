"""The N2 slice: the specimen, its cuts, and the checks of ADR 0005 section 4,
each with its negative control."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from agatesim.volume.config import SliceConfig

from nodule.specimen import FractureConfig, Specimen, SpecimenConfig, load_preset
from nodule.specimen_checks import (
    coincident_agreement,
    fold_check,
    overlap_counts,
    tip_offsets_mm,
)


@pytest.fixture(scope="module")
def config() -> SpecimenConfig:
    return load_preset("n2-slice")


@pytest.fixture(scope="module")
def specimen(config: SpecimenConfig) -> Specimen:
    return Specimen.build(config)


def _with_fracture(config: SpecimenConfig, **update: object) -> Specimen:
    fracture = config.fractures[0].model_copy(update=update)
    return Specimen.build(config.model_copy(update={"fractures": [fracture]}))


def test_preset_has_the_slice_parts(config: SpecimenConfig) -> None:
    assert config.volume.solid.preset == "nodule-irregular"  # lobed exterior
    assert len(config.volume.solid.filaments) == 2  # one inclusion type: tubes
    cap = float(getattr(config.volume.model, "wobble_max_slope", 0.0))
    assert 0 < cap < 1  # the wobble's slope is capped, so bands cannot fold
    assert len(config.fractures) == 1


def test_cut_covers_the_stone_on_world_fixed_pixels(specimen: Specimen) -> None:
    plane = specimen.plane(SliceConfig())
    cut = specimen.cut(plane, 0.5)
    assert cut.inside.shape == (len(cut.t), len(cut.s))
    assert cut.inside.mean() > 0.2
    border = np.concatenate([cut.inside[0], cut.inside[-1], cut.inside[:, 0], cut.inside[:, -1]])
    assert not border.any()  # the window holds the whole (fractured) stone
    k = cut.s / 0.5
    assert np.allclose(k, np.rint(k))  # pixel centres at multiples of the spacing
    assert np.isfinite(cut.fracture_mm).any()


def test_tubes_appear_in_a_cut_across_them(specimen: Specimen) -> None:
    # Filament cores carry no deposition time (agatesim reports them as Fe-silicates).
    across = specimen.plane(SliceConfig(tilt_deg=0, offset_mm=6))
    cut = specimen.cut(across, 0.1)
    core = cut.inside & np.isnan(cut.fields.t_dep)
    z = 6.0 + float(specimen.volume.solid.centre()[2])
    for row in specimen.volume.setting.tubes:
        a, b, r = row[:3], row[3:6], float(row[6])
        assert min(a[2], b[2]) < z < max(a[2], b[2])
        p = a + (z - a[2]) / (b[2] - a[2]) * (b - a)
        s = (p - across.origin) @ across.u
        t = (p - across.origin) @ across.v
        ss, tt = np.meshgrid(cut.s, cut.t)
        assert core[np.hypot(ss - s, tt - t) < r].any()
    area = np.count_nonzero(core) * 0.1**2
    expected = sum(np.pi * float(r) ** 2 for r in specimen.volume.setting.tubes[:, 6])
    assert 0.5 * expected < area < 2 * expected


def test_fracture_offsets_the_bands(specimen: Specimen) -> None:
    f = specimen.fractures[0]
    e0, e1, _ = f.frame
    a = np.linspace(-10, 10, 81)
    near = np.asarray(f.centre) + 0.05 * e0 + np.outer(a, e1)  # just on the moving side
    moved, _ = specimen.evaluate(near)
    still = specimen.volume.evaluate(near, specimen.volume.solid.contains(near))
    assert np.count_nonzero(moved.band != still.band) > 5
    assert np.abs(f.offset(near)).max() > 0.8 * f.slip_mm


# --- Check 1: bands do not fold ---------------------------------------------


def test_bands_do_not_fold(specimen: Specimen) -> None:
    result = fold_check(specimen)
    assert result.checked > 1000
    assert result.folds == 0


def test_fold_check_rejects_an_uncapped_steep_wobble(config: SpecimenConfig) -> None:
    model = config.volume.model.model_copy(
        update={"wobble_mm": 1.5, "wobble_scale_mm": 1.0, "wobble_max_slope": 0.0}
    )
    volume = config.volume.model_copy(update={"model": model})
    steep = Specimen.build(config.model_copy(update={"volume": volume}))
    assert fold_check(steep).folds > 0


# --- Check 2: offset zero at the tip line -------------------------------------


def test_fracture_offset_is_zero_at_its_tip(specimen: Specimen) -> None:
    assert all(v < 1e-9 for v in tip_offsets_mm(specimen))


def test_tip_check_rejects_constant_slip_ending_in_the_stone(config: SpecimenConfig) -> None:
    flat = _with_fracture(config, along="flat")
    assert tip_offsets_mm(flat)[0] > 0.5 * config.fractures[0].slip_mm


# --- Check 3: the fracture map is one-to-one -----------------------------------


def test_fracture_map_is_one_to_one(specimen: Specimen) -> None:
    assert overlap_counts(specimen) == [0]


def test_overlap_check_rejects_a_steep_single_step(config: SpecimenConfig) -> None:
    # Slip falls from 10 mm to 0 within 4 mm along the slip: one step overruns.
    steep = _with_fracture(config, slip_mm=10.0, radius_along_mm=4.0, steps=1)
    assert steep.fractures[0].step_bound > 1
    assert overlap_counts(steep)[0] > 0


# --- Check 4: preview and full cut agree where pixels coincide ------------------


def test_preview_agrees_with_full_cut(specimen: Specimen) -> None:
    plane = specimen.plane(SliceConfig(tilt_deg=70, azimuth_deg=200, offset_mm=2))
    fine = specimen.cut(plane, 0.25)
    preview = specimen.cut(plane, 0.5)
    result = coincident_agreement(fine, preview)
    assert result.compared > 0.95 * preview.inside.size  # its window is up to a pixel wider
    assert result.differing == 0


def test_agreement_rejects_noise_drawn_per_pixel(specimen: Specimen) -> None:
    plane = specimen.plane(SliceConfig())
    fine = specimen.cut(plane, 0.25)
    preview = specimen.cut(plane, 0.5)

    def per_pixel(cut_fields_t: np.ndarray, seed: int) -> np.ndarray:
        noise = np.random.default_rng(seed).normal(0, 0.01, cut_fields_t.shape)
        return (cut_fields_t + noise).astype(np.float32)

    noisy_fine = replace(fine, fields=replace(fine.fields, t_dep=per_pixel(fine.fields.t_dep, 0)))
    noisy_preview = replace(
        preview, fields=replace(preview.fields, t_dep=per_pixel(preview.fields.t_dep, 0))
    )
    result = coincident_agreement(noisy_fine, noisy_preview)
    assert result.differing > 0.5 * np.count_nonzero(preview.inside)


def test_fracture_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="slip_mm"):
        FractureConfig(
            normal=(1, 0, 0),
            slip_direction=(0, 0, 1),
            slip_mm=-1,
            radius_along_mm=1,
            radius_across_mm=1,
        )
