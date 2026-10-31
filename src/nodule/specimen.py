"""One simple nodule, complete in 3D and cut anywhere: the N2 vertical slice (ADR 0005).

The specimen is an evaluator, world points (mm) to fields (ADR 0005 section 1).
agatesim prepares the unfractured nodule: a lobed exterior (`Solid`), bands
from its grid-free growth model with a slope-capped wobble (agatesim ADR 0076),
and filaments (tubes) as the one inclusion type. Fractures are applied
backward, youngest first: a point is stepped back to where it sat before the
fractures, and the unfractured model is read there (ADR 0005 section 2;
agatesim `volume.fractures`). So bands, tubes and the rim all move with the rock.

A cut is a plane sampled on pixel centres fixed in plane coordinates: pixel
(i, j) of spacing h sits at (i h, j h) from the plane's origin. A preview at
spacing 2h therefore lands exactly on every other pixel of a cut at h, and the
two must agree there (CLAUDE.md, resolution independence; ADR 0005 section 4).

Every parameter here is effective (ADR 0002) unless calibrated: the solid's
size and shape, the growth model's settings and the fractures' slip, tip
radii and drag. Couplings: fractures act on everything after the growth (an
assumption, ADR 0005 section 2: band noise lives in pre-fracture coordinates,
after [@wellmann2018]); a fracture has no aperture (assumption, same section).
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, fields, replace
from functools import cached_property
from importlib import resources
from typing import Literal

import numpy as np
from agatesim import progress
from agatesim.fields import Fields
from agatesim.grid import FloatArray
from agatesim.volume.config import SliceConfig, VolumeConfig
from agatesim.volume.fractures import BELL, FLAT, Fracture, evaluate_fractured
from agatesim.volume.simulate import VolumeSimulation, prepare_volume
from agatesim.volume.slicing import Plane, reshape_fields
from agatesim.volume.solids import BoolArray
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

CHUNK_POINTS = 250_000  # points per evaluation call; bounds peak memory
PROFILES = {"bell": BELL, "flat": FLAT}


class FractureConfig(BaseModel):
    """One flat fracture with slip (agatesim `Fracture`). Lengths in mm; all
    values are effective until calibrated against offsets on real stones."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    centre_mm: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="Centre, from the solid's centre."
    )
    normal: tuple[float, float, float] = Field(description="Points to the side that moves.")
    slip_direction: tuple[float, float, float] = Field(description="Projected onto the plane.")
    slip_mm: float = Field(ge=0, description="Largest slip (v_max), effective.")
    radius_along_mm: float = Field(gt=0, description="Tip ellipse along the slip, effective.")
    radius_across_mm: float = Field(gt=0, description="Tip ellipse across the slip, effective.")
    drag_mm: float | None = Field(
        default=None, gt=0, description="Width over which slip fades from the plane; none = block."
    )
    along: Literal["bell", "flat"] = "bell"  # flat never reaches 0 at the tip: a negative control
    across: Literal["bell", "flat"] = "bell"
    steps: int | None = Field(default=None, ge=1, description="Backward steps; none = automatic.")

    def build(self, solid_centre: FloatArray) -> Fracture:
        centre = np.asarray(solid_centre, dtype=np.float64) + np.asarray(self.centre_mm)
        return Fracture(
            centre=(float(centre[0]), float(centre[1]), float(centre[2])),
            normal=self.normal,
            slip_direction=self.slip_direction,
            slip_mm=self.slip_mm,
            radius_along_mm=self.radius_along_mm,
            radius_across_mm=self.radius_across_mm,
            drag_mm=math.inf if self.drag_mm is None else self.drag_mm,
            along=PROFILES[self.along],
            across=PROFILES[self.across],
            steps=self.steps,
        )


class SpecimenConfig(BaseModel):
    """The nodule (an agatesim 3D run) and its fractures, oldest first."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    volume: VolumeConfig
    fractures: list[FractureConfig] = Field(default_factory=list[FractureConfig])


def load_preset(name: str) -> SpecimenConfig:
    text = resources.files("nodule.presets").joinpath(f"{name}.toml").read_text("utf-8")
    return SpecimenConfig.model_validate(tomllib.loads(text))


@dataclass(frozen=True, eq=False)
class SpecimenCut:
    """A planar cut: fields on pixel centres (j, i) at plane coordinates (s[i], t[j])."""

    plane: Plane
    spacing_mm: float
    s: FloatArray  # (nx,) mm along plane.u
    t: FloatArray  # (ny,) mm along plane.v
    fields: Fields  # arrays shaped (ny, nx)
    inside: BoolArray  # (ny, nx): in the (fractured) stone
    fracture_mm: FloatArray  # (ny, nx): distance to the nearest fracture sheet; inf if none


@dataclass(frozen=True, eq=False)
class Specimen:
    volume: VolumeSimulation
    fractures: tuple[Fracture, ...]

    @classmethod
    def build(cls, config: SpecimenConfig) -> Specimen:
        volume = prepare_volume(config.volume)
        centre = volume.solid.centre()
        return cls(volume=volume, fractures=tuple(f.build(centre) for f in config.fractures))

    @cached_property
    def max_slip_mm(self) -> float:
        return sum(f.slip_mm for f in self.fractures)

    def evaluate(self, points: FloatArray) -> tuple[Fields, BoolArray]:
        """Fields at world points (n, 3), and which points lie in the stone."""
        return evaluate_fractured(
            self.volume.evaluate, self.volume.solid.contains, points, self.fractures
        )

    def fracture_distance(self, points: FloatArray) -> FloatArray:
        """Distance (mm) to the nearest fracture sheet, in present coordinates.

        Each sheet is placed where it formed; a younger fracture's slip moves an
        older sheet, which this ignores (one fracture in the slice)."""
        out = np.full(len(points), np.inf)
        for f in self.fractures:
            out = np.minimum(out, f.sheet_distance(points))
        return out

    def plane(self, cut: SliceConfig) -> Plane:
        return Plane.of(cut, self.volume.solid.centre())

    def window(self, plane: Plane, spacing_mm: float) -> tuple[FloatArray, FloatArray]:
        """Pixel centres (s, t) covering the stone's bounding box in the plane,
        padded by the total slip, at integer multiples of the spacing."""
        lo, hi = self.volume.solid.bounds()
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])  # fmt: skip
        rel = corners - plane.origin
        pad = self.max_slip_mm + spacing_mm
        axes: list[FloatArray] = []
        for d in (plane.u, plane.v):
            c = rel @ d
            k0 = math.floor((float(c.min()) - pad) / spacing_mm)
            k1 = math.ceil((float(c.max()) + pad) / spacing_mm)
            axes.append(np.arange(k0, k1 + 1) * spacing_mm)
        return axes[0], axes[1]

    def cut(self, plane: Plane, spacing_mm: float) -> SpecimenCut:
        if not spacing_mm > 0:
            raise ValueError("spacing_mm must be positive")
        s, t = self.window(plane, spacing_mm)
        ss, tt = np.meshgrid(s, t)
        points = plane.points(ss, tt)
        parts: list[tuple[Fields, BoolArray]] = []
        n = len(points)
        for start in range(0, n, CHUNK_POINTS):
            progress.report(start / n, "cutting")
            parts.append(self.evaluate(points[start : start + CHUNK_POINTS]))
        progress.report(1.0, "cutting")
        fields = _concat([p[0] for p in parts])
        inside = np.concatenate([p[1] for p in parts])
        return SpecimenCut(
            plane=plane,
            spacing_mm=spacing_mm,
            s=s,
            t=t,
            fields=reshape_fields(fields, ss.shape),
            inside=inside.reshape(ss.shape),
            fracture_mm=self.fracture_distance(points).reshape(ss.shape),
        )


def diagnostic_image(cut: SpecimenCut) -> NDArray[np.uint8]:
    """A false-colour picture for checking geometry, not a rendering: alternate
    bands light and dark, shaded by growth time; tube cores green; the stone's
    surroundings brown; fracture sheets red. Rows run top to bottom (+v up)."""
    f = cut.fields
    t = np.nan_to_num(f.t_dep, nan=0.0)
    grey = 0.3 + 0.45 * (f.band % 2) + 0.25 * t
    rgb = np.repeat(grey[..., None], 3, axis=2)
    rgb[cut.inside & np.isnan(f.t_dep)] = (0.25, 0.55, 0.25)
    rgb[~cut.inside] = (0.25, 0.2, 0.15)
    rgb[cut.fracture_mm < 0.75 * cut.spacing_mm] = (0.85, 0.1, 0.1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[::-1]


def main(argv: list[str] | None = None) -> None:
    """Cut a preset, report time and peak memory, save a diagnostic picture."""
    import argparse
    import time
    import tracemalloc
    from pathlib import Path

    from PIL import Image

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--preset", default="n2-slice")
    parser.add_argument("--spacing", type=float, default=0.1, help="pixel spacing, mm")
    parser.add_argument("--tilt", type=float, default=90.0, help="normal's tilt from vertical")
    parser.add_argument("--azimuth", type=float, default=270.0, help="normal's direction")
    parser.add_argument("--offset", type=float, default=0.0, help="mm from the centre")
    parser.add_argument("--out", type=Path, default=Path("runs/specimen"))
    args = parser.parse_args(argv)

    tracemalloc.start()
    t0 = time.perf_counter()
    specimen = Specimen.build(load_preset(args.preset))
    t1 = time.perf_counter()
    cut_config = SliceConfig(tilt_deg=args.tilt, azimuth_deg=args.azimuth, offset_mm=args.offset)
    cut = specimen.cut(specimen.plane(cut_config), args.spacing)
    t2 = time.perf_counter()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    n_in = int(cut.inside.sum())
    print(f"prepare {t1 - t0:.2f} s; cut {cut.inside.shape[1]} x {cut.inside.shape[0]} px "
          f"at {args.spacing} mm, {n_in} in the stone: {t2 - t1:.2f} s "
          f"({(t2 - t1) / max(n_in, 1) * 1e6:.1f} s per 10^6 stone points); "
          f"peak traced memory {peak / 1e6:.0f} MB")  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"{args.preset}-t{args.tilt:g}-a{args.azimuth:g}-o{args.offset:g}-{args.spacing:g}mm.png"
    Image.fromarray(diagnostic_image(cut)).save(args.out / name)
    print(f"saved {args.out / name}")


def _concat(parts: list[Fields]) -> Fields:
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    update: dict[str, object] = {}
    for f in fields(first):
        values = [getattr(p, f.name) for p in parts]
        update[f.name] = None if values[0] is None else np.concatenate(values)
    return replace(first, **update)


if __name__ == "__main__":
    main()
