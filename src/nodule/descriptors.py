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

**Two image paths.** The stand-in, not a rendering: chalcedony takes
agatesim's colour from its translucency and staining
(`render.chalcedony_colour`), without the glow, sub-banding or rind of
`render.render_slab`; tubes and anything else that isn't chalcedony are drawn
dark, as voids. It sizes every cut, and its values are the point of
reference ADR 0008 section 3 reports. The rendered path (`rendering`, GPU)
keeps the stand-in's size and replaces only the final image by a render
under the stratum's light (`nodule.render.cutface`; ADR 0008 section 2),
from the stone's band record or a fake's (`rendering_for`). F5 measures the
render as it is, without the photo's blur and noise. Appearance only: never
evidence for a formation mechanism (ADR 0002). Every parameter here is
`effective`.

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
from typing import Any

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
from nodule.compare import STRATA, library_table
from nodule.render import fakes
from nodule.render.cutface import MOST_SPP, CutRender, Rendering, render_cut
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
    rendered: CutRender | None = None  # the rendered path's record; None for the stand-in


def photo_cut(
    specimen: Specimen,
    plane: Plane,
    look: Look,
    most: int = MOST_PX,
    rendering: Rendering | None = None,
    seed: int = 1,
) -> PhotoCut:
    """The cut at the spacing that shows its bands as `look`'s photo shows its
    own. With `rendering`, the final image is rendered (GPU; ADR 0008
    section 2: the size still comes from the stand-in probe)."""
    probe_mm = extent_mm(specimen, plane) / PROBE_PX
    cut = specimen.cut(plane, probe_mm)
    rgb = stand_in_rgb(cut)
    pixels = resolution_for(rgb, cut.inside, look, most)
    if pixels != PROBE_PX:
        cut = specimen.cut(plane, probe_mm * PROBE_PX / pixels)
    if rendering is None:
        return PhotoCut(cut, rgb if pixels == PROBE_PX else stand_in_rgb(cut), pixels)
    r = render_cut(specimen, cut, rendering, look.noise, seed)
    return PhotoCut(cut, r.rgb, pixels, r)


def undegraded(look: Look) -> Look:
    """F5's look: no blur and no noise, the photo's spacing kept (ADR 0008 amendment 4)."""
    return replace(look, edge_per_spacing=0.0, noise=0.0)


def measure(
    pc: PhotoCut, look: Look, rng: np.random.Generator, *, degrade: bool = True
) -> dict[str, object]:
    """C6-C17 of a cut made to look like `look`'s photo, with
    `measure_cut`'s record of the blur (module docstring). Values are None
    where they could not be judged; `bands` is False where the gate found no
    bands, before or after the blur. `degrade=False` is F5: measured as
    rendered, `matched` None."""
    shape_rng, noise_rng, band_rng = rng.spawn(3)
    look = look if degrade else undegraded(look)
    mask = pc.cut.inside
    quartz = pc.cut.fields.phase == Phase.QUARTZ
    entry = measure_cut(pc.rgb, mask, quartz, look, shape_rng)
    if not degrade and "matched" in entry:
        entry["matched"] = None
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
    rendering: Rendering | None = None,
) -> list[dict[str, object]]:
    """`cuts` random cuts (normals uniform over the sphere, as agatesim's
    calibration cuts), each made to look like a photo drawn at random.
    With `rendering`, each cut's image is rendered (GPU), and F5 is
    `rendering.undegraded`. The first three RNG streams are the stand-in's, so
    a rendered run cuts the same planes and draws the same photos."""
    cut_rng, look_rng, measure_rng, render_rng = np.random.default_rng(seed).spawn(4)
    out: list[dict[str, object]] = []
    for k, spec in enumerate(random_cuts(specimen.volume.solid, cuts, cut_rng, PROBE_PX)):
        progress.report(k / cuts, f"cut {k + 1} of {cuts}")
        look = photo_looks[int(look_rng.integers(len(photo_looks)))]
        plane = specimen.plane(spec)
        render_seed = int(render_rng.integers(1, 2**32 - 2))  # the kernel's seed is uint32
        pc = photo_cut(specimen, plane, look, most, rendering, render_seed)
        where = [round(spec.tilt_deg, 2), round(spec.azimuth_deg, 2), round(spec.offset_mm, 2)]
        entry: dict[str, object] = {"cut": k, "plane": where, "pixels": pc.pixels}
        if pc.rendered is not None:
            r = pc.rendered
            entry |= {"spp": r.spp, "render_noise": round(r.noise, 3), "capped": r.capped,
                      "fill_s": round(r.fill_s, 2), "render_s": round(r.render_s, 2)}  # fmt: skip
        degrade = rendering is None or not rendering.undegraded
        entry |= measure(pc, look, measure_rng.spawn(1)[0], degrade=degrade)
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


RUNS = ("stand-in", "stone", "F1", "F2", "F3", "F4", "F5")  # ADR 0008 section 4


def rendering_for(
    specimen: Specimen,
    run: str,
    stratum: str,
    rng: np.random.Generator,
    points: int = fakes.STONE_POINTS,
    **kw: Any,
) -> Rendering:
    """The rendering of a rendered run: the stone's record, read once from the
    whole stone, or a fake of it (ADR 0008 amendment 4). The record comes from
    `rng`'s first spawned stream and F4's phases from its second, so the stone
    and every fake share one record for a given `rng`. Reflected faces take L1,
    slabs L3 (ADR 0008 7c)."""
    record_rng, fake_rng = rng.spawn(2)
    record = fakes.stone_record(specimen, record_rng, points)
    if run == "F1":
        record = fakes.smoothed(record)
    elif run == "F2":
        record = fakes.remapped(record)
    elif run == "F3":
        record = fakes.regular(record)
    elif run == "F4":
        record = fakes.phase_randomised(record, fake_rng)
    elif run not in ("stone", "F5"):
        raise ValueError(f"no rendered run {run!r}")
    make = Rendering.reflected if stratum == "reflected" else Rendering.transmitted
    return make(record, undegraded=run == "F5", **kw)


def stratum_looks(photo_looks: list[Look], stratum: str) -> list[Look]:
    """The looks of one stratum's compared photos (`compare.library_table`)."""
    ids = set(library_table(stratum).ids)
    return [lk for lk in photo_looks if lk.photo in ids]


def main(argv: list[str] | None = None) -> None:
    """Measure random cuts of a preset; print each, and the set beside the
    library's spread. Reports time and peak traced memory. `--run` other than
    stand-in renders each cut (GPU; run as `uv run --no-sync`), with looks from
    `--stratum`'s photos only."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--preset", default="n2-slice")
    parser.add_argument("--cuts", type=int, default=8)
    parser.add_argument("--most", type=int, default=MOST_PX, help="most pixels across a cut")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/descriptors"))
    parser.add_argument("--run", default="stand-in", choices=RUNS)
    parser.add_argument("--stratum", default="reflected", choices=sorted(STRATA))
    parser.add_argument("--record-seed", type=int, default=0, help="the stone record's stream")
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--depth-mm", type=float, default=None, help="default: by light")
    parser.add_argument("--most-spp", type=int, default=MOST_SPP)
    args = parser.parse_args(argv)

    photo_looks, entries = library_looks()
    tracemalloc.start()
    t0 = time.perf_counter()
    specimen = Specimen.build(load_preset(args.preset))
    rendering = None
    if args.run != "stand-in":
        photo_looks = stratum_looks(photo_looks, args.stratum)
        kw: dict[str, Any] = {"exposure": args.exposure, "most_spp": args.most_spp}
        if args.depth_mm is not None:
            kw["depth_mm"] = args.depth_mm
        rng = np.random.default_rng(args.record_seed)
        rendering = rendering_for(specimen, args.run, args.stratum, rng, **kw)
    rows = survey(
        specimen, photo_looks, args.cuts, seed=args.seed, most=args.most, rendering=rendering
    )
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
    if rendering is not None:
        from nodule.render import _gpu

        pool = _gpu.cupy().get_default_memory_pool()
        print(f"GPU memory pool held at the end: {pool.total_bytes() / 1e6:.0f} MB")
        for r in rows:
            print(f"cut {r['cut']}: {r['pixels']} px, {r['spp']} spp, render noise "
                  f"{r['render_noise']} (capped {r['capped']}), fill {r['fill_s']} s, "
                  f"render {r['render_s']} s")  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "" if rendering is None else f"-{args.run}-{args.stratum}"
    path = args.out / f"{args.preset}{tag}-{args.cuts}cuts-seed{args.seed}-{args.most}px.json"
    path.write_text(json.dumps(rows, indent=1), "utf-8")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
