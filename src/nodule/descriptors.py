"""Band descriptors of the specimen's cuts, measured by the library's own code
(N2 step 3b; ADR 0005, ADR 0006 section 10).

ADR 0006's fallback compares synthetic stones with the library by image
statistics at set level, so a cut must be measured exactly as a photo is.
Each cut goes through the steps agatesim's calibration uses for its own 3D
models (agatesim ADR 0069; `agatesim.cut_shapes`), made to look like one
library photo picked at random (its `Look`):

- **size**: a first look at `PROBE_PX` pixels across finds the cut's band
  spacing, and the cut is redone so its bands show as the photo shows its own
  (`cut_shapes.resolution_for`, at most `most` pixels across).
- **blur and noise**: blurred until its edge width over its band spacing is
  the photo's, then grey noise of the photo's level is added
  (`cut_shapes.measure_cut`).
- **measure**: C7, C8, C9 and C11 by `agatesim.specimen.shape`, as for the
  photos in agatesim's cache, and C6 and C10 by `nodule.bands`, as
  `nodule.tags` measures the photos, after the quality gate finds the noise
  and band spacing of the photo-like cut. `measure_cut` keeps its blurred
  image to itself, so C6 and C10 read a second blurred copy, at the same blur
  and noise level but its own noise draw. As for the photos, C6, C10 and
  C12-C16 are measured only where the shape gate passes the face. C12 (from
  C6's profiles), C13 (`nodule.concentric`), C14 (`nodule.defects`), C15
  (`nodule.curvature`) and C16 (`nodule.orientation`) read the same image as
  C6 and C10.

**The image is a stand-in, not a rendering.** Chalcedony takes agatesim's
colour from its translucency and staining (`render.chalcedony_colour`),
without the glow, sub-banding or rind of `render.render_slab`; tubes and
anything else that isn't chalcedony are drawn dark, as voids. The renderer is
N2 step 4 and needs its own ADR, so values measured here show that the code
runs on cuts and that its controls hold; they are not yet a comparison with
the library. Appearance only: never evidence for a formation mechanism
(ADR 0002). Every parameter here is `effective`.

The pixel noise is drawn per pixel of the cut, as a photo's is per pixel of
its camera: it belongs to the look, not to the stone (as in agatesim). The
stone's own fields are resolution-independent (`nodule.specimen`).
"""

from __future__ import annotations

import argparse
import json
import math
import time
import tracemalloc
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from agatesim import progress
from agatesim.cut_shapes import (
    PROBE_RESOLUTION,
    Look,
    looks,
    measure_cut,
    photo_like,
    resolution_for,
)
from agatesim.fields import Phase
from agatesim.render import VOID, chalcedony_colour
from agatesim.shapes import spreads
from agatesim.specimen.quality import assess, luminance
from agatesim.specimen.segment import trace_outline
from agatesim.volume.slicing import Plane, random_cuts
from numpy.typing import NDArray

from nodule import bands
from nodule.specimen import Specimen, SpecimenCut, load_preset
from nodule.tags import (
    FROM_CONTINUITY,
    FROM_CURVE,
    FROM_DEFECTS,
    FROM_ORDER,
    FROM_RINGS,
    FROM_SPECTRUM,
    GATE_PASSES,
    IMPORTED,
    agatesim_dir,
    face_measure,
)

PROBE_PX = PROBE_RESOLUTION  # pixels across the first look at a cut
MOST_PX = 2048  # most pixels across a measured cut (cost: see `main`)
# The descriptors reported, with where they come from (C12-C16 as the library
# tags name them, read on the same image as C6 and C10).
FROM_SHAPE = dict(IMPORTED)  # C7-C9, C11 and C17, named as in the library tags
FROM_BANDS = {"C6_count": "count", "C10_persistence": "persistence"}
LATER = (
    *FROM_CONTINUITY,
    *FROM_RINGS,
    *FROM_ORDER,
    *FROM_CURVE,
    *FROM_DEFECTS,
    *FROM_SPECTRUM,
)  # C12-C16 and C26's summaries


def stand_in_rgb(cut: SpecimenCut) -> NDArray[np.uint8]:
    """The cut's stand-in image (module docstring), rows as the cut's arrays."""
    f = cut.fields
    rgb = np.empty((*cut.inside.shape, 3), dtype=np.float64)
    rgb[:] = VOID
    chal = cut.inside & (f.phase == Phase.CHALCEDONY) & np.isfinite(f.translucency)
    trans = np.nan_to_num(f.translucency, nan=0.0).astype(np.float64)
    iron = np.nan_to_num(f.impurity, nan=0.0).astype(np.float64)
    colour = chalcedony_colour(trans, iron, f.pigment)
    rgb[chal] = colour[chal]
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def extent_mm(specimen: Specimen, plane: Plane) -> float:
    """The larger side of the cut's window (mm)."""
    s, t = specimen.window(plane, 1.0)
    return float(max(s[-1] - s[0], t[-1] - t[0]))


@dataclass(frozen=True)
class PhotoCut:
    cut: SpecimenCut
    rgb: NDArray[np.uint8]
    pixels: int  # pixels across, as `resolution_for` asked


def photo_cut(specimen: Specimen, plane: Plane, look: Look, most: int = MOST_PX) -> PhotoCut:
    """The cut at the spacing that shows its bands as `look`'s photo shows its own."""
    probe_mm = extent_mm(specimen, plane) / PROBE_PX
    cut = specimen.cut(plane, probe_mm)
    rgb = stand_in_rgb(cut)
    pixels = resolution_for(rgb, cut.inside, look, most)
    if pixels != PROBE_PX:
        cut = specimen.cut(plane, probe_mm * PROBE_PX / pixels)
        rgb = stand_in_rgb(cut)
    return PhotoCut(cut, rgb, pixels)


def measure(pc: PhotoCut, look: Look, rng: np.random.Generator) -> dict[str, object]:
    """C6-C17 of a cut made to look like `look`'s photo, with
    `measure_cut`'s record of the blur (module docstring). Values are None
    where they could not be judged; `bands` is False where the gate found no
    bands, before or after the blur."""
    shape_rng, noise_rng, band_rng = rng.spawn(3)
    mask = pc.cut.inside
    quartz = pc.cut.fields.phase == Phase.QUARTZ
    entry = measure_cut(pc.rgb, mask, quartz, look, shape_rng)
    out: dict[str, object] = {k: entry.get(k) for k in ("photo", "level", "bands", "matched")}
    out |= {"sigma_px": entry.get("sigma_px"), "spacing_px": entry.get("spacing_px")}
    out |= {name: entry.get(there) for name, there in FROM_SHAPE.items()}
    out |= {name: None for name in (*FROM_BANDS, *LATER)}
    sigma = entry.get("sigma_px")
    if not entry.get("bands") or not isinstance(sigma, int | float):
        return out
    if entry.get("level") not in GATE_PASSES:  # as `nodule.tags.passes` for photos
        return out
    seen = photo_like(pc.rgb, float(sigma), look.noise, noise_rng)
    q = assess(seen, mask, trace_outline(mask))
    if q.band_spacing_px is None:
        return out
    lum = luminance(seen)
    b = bands.measure(lum, mask, 1.0, q.noise, q.band_spacing_px, rng=band_rng, ladder=False)
    f = face_measure(lum, mask, 1.0, q.noise, q.band_spacing_px)
    pairs = (
        (b, FROM_BANDS),
        (b, FROM_CONTINUITY),
        (b.spectrum, FROM_SPECTRUM),
        (f.rings, FROM_RINGS),
        (f.order, FROM_ORDER),
        (f.curve, FROM_CURVE),
        (f.defects, FROM_DEFECTS),
    )
    for source, names in pairs:
        for name, field in names.items():
            value = getattr(source, field) if source is not None else None
            if isinstance(value, bool | int) or value is None:
                out[name] = value
            else:
                out[name] = round(float(value), 4)
    return out


def survey(
    specimen: Specimen,
    photo_looks: list[Look],
    cuts: int,
    *,
    seed: int = 0,
    most: int = MOST_PX,
) -> list[dict[str, object]]:
    """`cuts` random cuts (normals uniform over the sphere, as agatesim's
    calibration cuts), each made to look like a photo drawn at random."""
    cut_rng, look_rng, measure_rng = np.random.default_rng(seed).spawn(3)
    out: list[dict[str, object]] = []
    for k, spec in enumerate(random_cuts(specimen.volume.solid, cuts, cut_rng, PROBE_PX)):
        progress.report(k / cuts, f"cut {k + 1} of {cuts}")
        look = photo_looks[int(look_rng.integers(len(photo_looks)))]
        plane = specimen.plane(spec)
        pc = photo_cut(specimen, plane, look, most)
        where = [round(spec.tilt_deg, 2), round(spec.azimuth_deg, 2), round(spec.offset_mm, 2)]
        entry: dict[str, object] = {"cut": k, "plane": where, "pixels": pc.pixels}
        entry |= measure(pc, look, measure_rng.spawn(1)[0])
        out.append(entry)
    progress.report(1.0, "cuts measured")
    return out


def library_looks(agatesim: Path | None = None) -> tuple[list[Look], dict[str, dict[str, object]]]:
    """The library photos' looks and agatesim's cached statistics for them."""
    path = (agatesim or agatesim_dir()) / "references" / "models" / "shapes.json"
    entries: dict[str, dict[str, object]] = json.loads(path.read_text("utf-8"))
    return looks(entries), entries


def bandless(cut: SpecimenCut) -> SpecimenCut:
    """Negative control: the same cut with even translucency and staining, so
    the face holds no bands."""
    f = cut.fields
    even = replace(
        f,
        translucency=np.where(np.isfinite(f.translucency), 0.5, np.nan).astype(np.float32),
        impurity=np.where(np.isfinite(f.impurity), 0.1, np.nan).astype(np.float32),
    )
    return replace(cut, fields=even)


def _median(values: list[object]) -> float | None:
    xs = [float(v) for v in values if isinstance(v, int | float) and math.isfinite(float(v))]
    return float(np.median(xs)) if xs else None


def main(argv: list[str] | None = None) -> None:
    """Measure random cuts of a preset; print each, and the set beside the
    library's spread. Reports time and peak traced memory."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--preset", default="n2-slice")
    parser.add_argument("--cuts", type=int, default=8)
    parser.add_argument("--most", type=int, default=MOST_PX, help="most pixels across a cut")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/descriptors"))
    args = parser.parse_args(argv)

    photo_looks, entries = library_looks()
    tracemalloc.start()
    t0 = time.perf_counter()
    specimen = Specimen.build(load_preset(args.preset))
    rows = survey(specimen, photo_looks, args.cuts, seed=args.seed, most=args.most)
    elapsed = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    names = [*FROM_SHAPE, *FROM_BANDS, *LATER]
    for r in rows:
        values = " ".join(f"{n if n in LATER else n.split('_')[0]}={r[n]}" for n in names)
        print(f"cut {r['cut']} {r['pixels']} px, {r['level']}, bands={r['bands']}: {values}")
    library = spreads(entries, ("fit", "borderline"))
    print("descriptor: cuts median | library 10th / median / 90th percentile")
    for name, there in FROM_SHAPE.items():
        lib = library.get(there)
        span = f"{lib.low:.3f} / {lib.median:.3f} / {lib.high:.3f} (n={lib.n})" if lib else "none"
        print(f"  {name}: {_median([r[name] for r in rows])} | {span}")
    for name in FROM_BANDS:
        print(f"  {name}: {_median([r[name] for r in rows])} | see references/library-tags.toml")
    for name in LATER:
        if name != "C16_corr_censored":  # library: after the tag rebuild of step 2
            print(f"  {name}: {_median([r[name] for r in rows])} | library not yet measured")
    print(f"{args.cuts} cuts: {elapsed:.1f} s; peak traced memory {peak / 1e6:.0f} MB")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.preset}-{args.cuts}cuts-seed{args.seed}-{args.most}px.json"
    path.write_text(json.dumps(rows, indent=1), "utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
