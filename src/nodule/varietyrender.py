"""Renders of a variety sheet's stones with their layers (ADR 0019): the first look at
ADR 0018's features through the path tracer instead of the stand-in colours.

Per stone the renderer can show (no open cavity yet: ADR 0019 option for the owner),
two views of each cut, on one grid fill (owner's choice, ADR 0019 B3):

- **the look**: the dome (L1, ADR 0008 7c; black bottom, ADR 0014) without the face's
  mirror reflection, at the photo compare's mean lightness (`Rendering.look`);
- **a backlit slice** SLICE_MM thick on a light box (L3, `Rendering.transmitted`), at
  one fixed exposure so slices differ in how much light they pass.

The whole face, and for a dusty stone a 12 mm close-up at 0.03 mm pixels where its dust
is densest, each beside its stand-in preview. No photo blur or noise: these are for the
eye, not the blind test.

Run: `uv run --no-sync python -m nodule.varietyrender --seed 1 --out runs/variety-render`.
GPU. Logs time per stone; resumes, skipping stones already saved.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from agatesim import progress
from PIL import Image, ImageDraw

from nodule.render.cutface import CutRenderer, Rendering
from nodule.render.fakes import stone_record
from nodule.render.materials import OpticsTable
from nodule.render.record import BandRecord
from nodule.sheet import dustiest, preview_rgb, window_cut
from nodule.specimen import Specimen, SpecimenCut
from nodule.variety import StoneRecipe, build, cut_of, draw_sheet

WHOLE_PX = 600  # pixels across the stone's box
WHOLE_GRID_MM = 0.1
CLOSE_MM, CLOSE_PIXEL_MM, CLOSE_GRID_MM = 12.0, 0.03, 0.03
SPP = 512  # samples per pixel (effective; noise falls as 1/sqrt(spp))
SLICE_MM = 3.0  # backlit slice thickness, owner's choice (2026-11-09; and ADR 0007 am. 5)
# Light box exposure (transmittance factor x this), `effective`, one for every stone so
# slices differ in brightness as they pass more or less light (2026-11-09 look sheets).
BACKLIT_EXPOSURE = 3.0
VIEWS = ("look", "backlit")
# Hematite pigment's red absorption and carnelian's loading (owner's pick from the
# 2026-11-09 carnelian sweeps, ADR 0019 s.6): `effective`, *speculative*.
HEMATITE_RED_K = 0.05
CARNELIAN_LOADING = 0.5


def optics_for(r: StoneRecipe, base: OpticsTable | None = None) -> OpticsTable:
    """The optics a variety stone renders with: `base` (default OpticsTable()) with the
    hematite pigment's red absorption at HEMATITE_RED_K, and carnelian's iron-oxide
    loading x CARNELIAN_LOADING (its slices were too opaque, owner 2026-11-09)."""
    base = base or OpticsTable()
    phi = base.phi_iron_oxide * (CARNELIAN_LOADING if r.colour == "carnelian" else 1.0)
    return replace(base, hematite_red_k=HEMATITE_RED_K, phi_iron_oxide=phi)


def renderable(r: StoneRecipe) -> bool:
    """The GPU has no void material yet (ADR 0019): no open cavity."""
    return r.centre != "quartz-lined"


def render_stone(
    r: StoneRecipe,
    out: Path,
    spp: int = SPP,
    optics: OpticsTable | None = None,
    close_ups: bool = True,
) -> dict[str, object]:
    optics = optics_for(r, optics)
    t0 = time.perf_counter()
    specimen, _ = build(r)
    lo, hi = specimen.volume.solid.bounds()
    spacing = float(np.max(np.asarray(hi) - np.asarray(lo))) / WHOLE_PX
    cut = cut_of(specimen, r, spacing)
    record = stone_record(specimen, np.random.default_rng(r.seed))
    rng = np.random.default_rng(r.seed + 1)
    row: dict[str, object] = {"stone": r.index + 1, "summary": r.summary()}
    name = f"stone-{r.index + 1:02d}"
    row |= views(specimen, cut, record, WHOLE_GRID_MM, optics, spp, rng, out / name, "whole")
    Image.fromarray(preview_rgb(cut, r)).save(out / f"{name}-preview.png")
    if r.dust is not None and close_ups:
        near = window_cut(specimen, cut, dustiest(cut, CLOSE_MM), CLOSE_MM, CLOSE_PIXEL_MM)
        row |= views(specimen, near, record, CLOSE_GRID_MM, optics, spp, rng,
                     out / f"{name}-close", "close")  # fmt: skip
        Image.fromarray(preview_rgb(near, r)).save(out / f"{name}-close-preview.png")
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def views(
    specimen: Specimen,
    cut: SpecimenCut,
    record: BandRecord,
    grid_mm: float,
    optics: OpticsTable,
    spp: int,
    rng: np.random.Generator,
    stem: Path,
    key: str,
) -> dict[str, object]:
    """The look and the backlit slice of one cut, on one grid fill; saves
    `<stem>-look.png` and `<stem>-backlit.png`."""
    look = CutRenderer(specimen, cut, Rendering.look(record, grid_mm=grid_mm, optics=optics),
                       rng, fill_depth_mm=SLICE_MM)  # fmt: skip
    back = look.on(Rendering.transmitted(record, grid_mm=grid_mm, depth_mm=SLICE_MM,
                                         optics=optics, exposure=BACKLIT_EXPOSURE))  # fmt: skip
    row: dict[str, object] = {f"{key}_fill_s": round(look.fill_s, 1)}
    for view, cr in zip(VIEWS, (look, back), strict=True):
        cr.more(spp)
        Image.fromarray(cr.rgb[::-1]).save(f"{stem}-{view}.png")
        row |= {f"{key}_{view}_s": round(cr.render_s, 1), f"{key}_{view}_noise": round(cr.noise, 2)}
    row[f"{key}_look_exposure"] = round(look.exposure, 2)
    return row


COLUMNS = {
    "look": "look (dome, no face glare)",
    "backlit": f"backlit {SLICE_MM:g} mm slice",
    "preview": "stand-in preview",
    "close-look": "12 mm look",
    "close-backlit": "12 mm backlit",
    "close-preview": "12 mm preview",
}


def contact_sheet(out: Path, rows: list[dict[str, object]], tile: int = 300) -> Path:
    """Look | backlit | preview, then the close-up's three where there is one."""
    head = 24
    sheet = Image.new("RGB", (len(COLUMNS) * tile, head + len(rows) * (tile + 22)),
                      (235, 235, 235))  # fmt: skip
    draw = ImageDraw.Draw(sheet)
    for k, label in enumerate(COLUMNS.values()):
        draw.text((k * tile + 6, 6), label, fill=(0, 0, 0))
    for n, row in enumerate(rows):
        y = head + n * (tile + 22)
        i = int(str(row["stone"]))
        names = list(COLUMNS)
        for k, name in enumerate(names):
            path = out / f"stone-{i:02d}-{name}.png"
            if not path.exists():
                continue
            im = Image.open(path)
            im.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            sheet.paste(im, (k * tile + (tile - im.width) // 2, y + (tile - im.height) // 2))
        draw.text((6, y + tile + 4), f"stone {i}. {row['summary']}", fill=(0, 0, 0))
    path = out / "render-sheet.png"
    sheet.save(path)
    return path


def main(argv: list[str] | None = None) -> None:
    """Render a variety sheet's stones (those the renderer can show)."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("runs/variety-render"))
    parser.add_argument("--spp", type=int, default=SPP)
    parser.add_argument(
        "--phi-scale", type=float, default=1.0, help="iron-oxide loading x this (all stones)"
    )
    parser.add_argument("--whole-only", action="store_true", help="no close-ups")
    parser.add_argument(
        "--stones", type=int, nargs="*", help="stone numbers (1-based); default all"
    )
    args = parser.parse_args(argv)
    out = args.out / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "log.jsonl"
    done: dict[int, dict[str, object]] = {}
    if log.exists():
        for line in log.read_text("utf-8").splitlines():
            row: dict[str, object] = json.loads(line)
            done[int(str(row["stone"]))] = row
    recipes = [r for r in draw_sheet(args.seed) if renderable(r)]
    if args.stones:
        recipes = [r for r in recipes if r.index + 1 in args.stones]
    for n, r in enumerate(recipes):
        progress.report(n / len(recipes), f"stone {r.index + 1}")
        if r.index + 1 in done:
            continue
        optics = OpticsTable(phi_iron_oxide=OpticsTable().phi_iron_oxide * args.phi_scale)
        row = render_stone(r, out, args.spp, optics, close_ups=not args.whole_only)
        done[r.index + 1] = row
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    rows = [done[k] for k in sorted(done)]
    print(f"saved {contact_sheet(out, rows)}")


if __name__ == "__main__":
    main()
