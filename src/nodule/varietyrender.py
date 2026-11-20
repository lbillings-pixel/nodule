"""Renders of a variety sheet's stones with their layers (ADR 0019): the first look at
ADR 0018's features through the path tracer instead of the stand-in colours.

Per stone, two views of each cut, on one grid fill (owner's choice, ADR 0019 B3); an
open cavity renders dark in the look and as a hole in the slice (ADR 0019 A1), except a
crystal-lined hollow in the whole face's look, drawn as air so its wall shows (ADR 0019
s.6.12 step 4; `--stand-in` keeps the dark stand-in there too):

- **the look**: the dome (L1, ADR 0008 7c) over a grey matte bottom (ADR 0019 s.6.5)
  5 mm down (s.6.10; close-ups 2 mm, CLOSE_LOOK_MM) without the face's mirror
  reflection, at the photo compare's mean lightness (`Rendering.look`);
- **a backlit slice** SLICE_MM thick on a light box (L3, `Rendering.transmitted`), at
  one fixed exposure so slices differ in how much light they pass, with a highlight
  roll-off (BACKLIT_KNEE) so clear bands keep their shading.

With `--sparkle`, a crystal-lined hollow's whole face also gets **the sparkle view**: the
look plus a small lamp on the same fill, its crystal faces glinting (SPARKLE_LAMP; ADR 0019
s.6.12 step 5).

The whole face, and for a dusty stone a 12 mm close-up at 0.03 mm pixels where its dust
is densest, each beside its stand-in preview. No photo blur or noise: these are for the
eye, not the blind test.

Run: `uv run --no-sync python -m nodule.varietyrender --seed 1 --out runs/variety-render`.
GPU. Logs time per stone; resumes, skipping stones already saved.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
from agatesim import progress
from PIL import Image, ImageDraw

from nodule.plumes import STAGE
from nodule.render.camera import Light
from nodule.render.cutface import LOOK_DEPTH_MM, CutRenderer, Rendering, sparkle_rgb
from nodule.render.fakes import stone_record
from nodule.render.materials import OpticsTable
from nodule.render.record import STEPS, BandRecord
from nodule.render.voidbox import VoidBox, build_void_box
from nodule.sheet import dressed_look, dustiest, preview_rgb
from nodule.specimen import Specimen, SpecimenCut, window_cut
from nodule.variety import StoneRecipe, build, cut_of, draw_sheet

WHOLE_PX = 600  # pixels across the stone's box
WHOLE_GRID_MM = 0.1
# White fine needles (0.15 mm wide) vanish on the 0.1 mm grid of a whole face; at 0.05 mm
# they show as faint strokes (work plan item 11, runs/needle-grid-probe/, 2026-11-17):
# `effective`. The fill costs ~7x (40 mm stone: 168 s, 3.9 GB RAM, 1.8 GB GPU), so
# the grid coarsens to stay within NEEDLE_VOXELS (~11 GB RAM, ~5 GB GPU, scaled from
# that stone, not measured; the largest needle stone on sheets 0-11 needs ~92 M at
# 0.05 mm).
NEEDLE_GRID_MM = 0.05
NEEDLE_VOXELS = 100e6
CLOSE_MM, CLOSE_PIXEL_MM, CLOSE_GRID_MM = 12.0, 0.03, 0.03
# Close-ups keep the 2 mm look (owner, 2026-11-18, ADR 0019 s.6.10): at 5 mm their grid,
# padded by the fill, would grow from ~36 M to ~90 M points (est. 8-9 GB RAM). `effective`
CLOSE_LOOK_MM = 2.0
SPP = 512  # samples per pixel (effective; noise falls as 1/sqrt(spp))
SLICE_MM = 3.0  # backlit slice thickness, owner's choice (2026-11-09; and ADR 0007 am. 5)
FILL_MM = max(SLICE_MM, LOOK_DEPTH_MM)  # one grid serves the look and the slice
# Light box exposure (transmittance factor x this), `effective`, one for every stone so
# slices differ in brightness as they pass more or less light (2026-11-09 look sheets).
BACKLIT_EXPOSURE = 3.0
# Highlight roll-off of the backlit view (owner's pick from runs/backlit-rolloff/,
# 2026-11-09, ADR 0019 s.6.6): `effective`; the bare light box stays white
BACKLIT_KNEE = 0.6
VIEWS = ("look", "backlit")
# The sparkle view (`--sparkle`; ADR 0019 s.6.12 step 5): the look plus a small lamp, for a
# crystal-lined hollow drawn as air, its faces glinting; the soft look stays the default
# (owner, 2026-11-19). The lamp: a disc at 45 deg from the normal toward the upper left, of
# SPARKLE_RADIUS_DEG and SPARKLE_SHARE of the dome's irradiance; glints only at the hollow's
# mouth (`Rendering.lamp`): counting scattered light there too gave mostly fireflies (4/4:
# 4.0 % of the hollow's pixels white, against 0.16 % for glints alone). All `effective`,
# chosen on 4/4 (runs/hollow-air/sparkle_probe.py).
SPARKLE_RADIUS_DEG = 5.0  # a desk lamp's shade from ~50 cm; 2.5 deg (L2) lit 1/4 the facets
SPARKLE_SHARE = 0.5
# The lamp pass's samples, as a share of the look's: a glint pixel saturates once any of its
# samples mirrors the lamp (R ~ 4.5 %: 99.7 % of glint pixels at 128 spp), and the lamp's
# light through the face is a third of the look's; a lamp pass costs ~8 looks a sample
# (4/4, 2026-11-20: 362 s against 44 s at 256 spp, probe set-up).
SPARKLE_SPP_SHARE = 0.25
SPARKLE_LAMP = Light("disc", theta_deg=45.0, phi_deg=135.0, irradiance=SPARKLE_SHARE * math.pi,
                     radius_deg=SPARKLE_RADIUS_DEG)  # fmt: skip
# Hematite pigment's red absorption and carnelian's loading (owner's pick from the
# 2026-11-09 carnelian sweeps, ADR 0019 s.6): `effective`, *speculative*.
HEMATITE_RED_K = 0.05
CARNELIAN_LOADING = 0.5
# Colour-centre absorption of a tinted quartz centre over clear quartz's, mm^-1 at the
# channels' 450, 550, 650 nm (work plan item 11): `effective`, *plausible*. Medium
# amethyst from a 6.2 mm Anahi slab's spectrum (o-ray, Rossman's Caltech data, not
# peer-reviewed), its 545 nm band within the range of a deep zone read off
# [@hassan1974, Fig. 3] (0.075, 0.165, 0.07), its shape matching a peer-reviewed
# spectrum (450/550 0.40, 650/550 0.31 off [@lehmann1967, Fig. 3], a stone ~6x deeper);
# medium smoky from a 15.5 mm Saline Valley spectrum (same server, read off its plot,
# (2o + e)/3), between [@marshall1955]'s moderately coloured stone and a dark one.
# Strength by the owner's eye.
QUARTZ_TINT_MUA: dict[str, tuple[float, float, float]] = {
    "amethyst": (0.033, 0.079, 0.022),
    "smoky": (0.12, 0.093, 0.072),
}
# x QUARTZ_TINT_MUA; the owner's pick 2x at the 2 mm look (2026-11-17, runs/item11-render/),
# lowered with the 5 mm look (owner, 2026-11-18, ADR 0019 s.6.10) so an amethyst centre
# keeps its 2 mm hue (green/red 0.79 on sheet 0 stone 11); smoky scaled alike (assumption)
TINT_STRENGTH = 0.8
# Milky hollow linings (ADR 0019 s.6.12): the recipe's level (0-1) to OpticsTable.milk_mus,
# all `effective`, the owner's look; untinted quartz-filled centres take the same scale,
# though one level reads whiter there, so their levels are drawn lower
# (recipe.FILLED_MILK_LOW_TO). Matched by lightness to the C-D they chose on 4/4
# (runs/hollow-air/milk_probe.py, 2026-11-20): the hollow's centre over the chalcedony
# face, per neutral milk_mus at the probe set-up (MILK_CURVE); D's centre is
# MILK_D_CENTRE (34.8 per mm; C's, 0.547, came at 13.1; at both the ring and rim agree
# within 2 %, so one strength matches the whole hollow). Level 0 is
# MILK_NEAR_CLEAR (an assumption: barely milky), 1 is D; levels in between are even in
# that lightness, so a uniform level spreads evenly to the eye on 4/4.
MILK_CURVE: tuple[tuple[float, float], ...] = (
    (1.0, 0.127), (2.0, 0.151), (4.0, 0.216), (8.0, 0.359), (16.0, 0.623), (32.0, 0.998),
    (40.0, 1.143), (48.0, 1.266),
)  # fmt: skip
MILK_D_CENTRE = 1.052
MILK_NEAR_CLEAR = 2.0
# White opal (ADR 0019 s.13): the recipe's level (0 faint, 1 solid) to the white row's
# transport mu_s (1 - g) at 550 nm, per mm, even in its logarithm between the probe sheet's
# faint and solid tiles, the range the owner chose (2026-11-20). `effective`.
WHITE_FAINT = 4.0
WHITE_SOLID = 16.0


def milk_mus(level: float) -> float:
    """A lining's milk_mus (per mm) for its recipe level (MILK_CURVE's comment)."""
    vs = np.log([v for v, _ in MILK_CURVE])
    cs = np.array([c for _, c in MILK_CURVE])
    lo = float(np.interp(math.log(MILK_NEAR_CLEAR), vs, cs))
    want = lo + min(max(level, 0.0), 1.0) * (MILK_D_CENTRE - lo)
    return float(np.exp(np.interp(want, cs, vs)))


def white_transport(level: float) -> float:
    """The white row's transport (per mm at 550 nm) for a recipe's white level."""
    return WHITE_FAINT * (WHITE_SOLID / WHITE_FAINT) ** min(max(level, 0.0), 1.0)


def optics_for(r: StoneRecipe, base: OpticsTable | None = None) -> OpticsTable:
    """The optics a variety stone renders with: `base` (default OpticsTable()) with the
    hematite pigment's red absorption at HEMATITE_RED_K, and carnelian's iron-oxide
    loading x CARNELIAN_LOADING (its slices were too opaque, owner 2026-11-09); a tinted
    centre's colour centres, and the milk (`milk_mus`) of a quartz-lined hollow's lining or
    an untinted quartz-filled centre (`StoneRecipe.takes_milk`), on quartz after the early
    stage (clear early crystals are quartz too and stay clear). Tinted filled centres keep
    clear quartz (item 16's look; owner, 2026-11-20, ADR 0019 s.6.12). White plumes or
    rims (ADR 0019 s.13) at `white_transport` of the recipe's level."""
    base = base or OpticsTable()
    phi = base.phi_iron_oxide * (CARNELIAN_LOADING if r.colour == "carnelian" else 1.0)
    out = replace(base, hematite_red_k=HEMATITE_RED_K, phi_iron_oxide=phi)
    first = math.ceil(STAGE * STEPS - 1e-9)
    if r.takes_milk:
        out = replace(out, milk_mus=milk_mus(r.milk), milk_from=first)
    if r.white != "none":
        out = replace(out, white=r.white, white_transport=white_transport(r.white_level))
    if r.tint == "none":
        return out
    mua = tuple(TINT_STRENGTH * v for v in QUARTZ_TINT_MUA[r.tint])
    return replace(out, quartz_tint_mua=mua, quartz_tint_from=first)  # pyright: ignore[reportArgumentType]


def whole_grid_mm(r: StoneRecipe, cut: SpecimenCut) -> float:
    """The whole face's grid spacing: WHOLE_GRID_MM, or for white fine needles
    NEEDLE_GRID_MM, coarsened (never past WHOLE_GRID_MM) so the grid under the face,
    padded by the slab, stays within NEEDLE_VOXELS. Hollow needles are left at
    WHOLE_GRID_MM (none rendered yet)."""
    c = r.crystals
    if c is None or c.habit != "fine needles" or c.fill != "white":
        return WHOLE_GRID_MM
    a = float(np.ptp(cut.s)) + 2 * FILL_MM
    b = float(np.ptp(cut.t)) + 2 * FILL_MM
    least = (a * b * FILL_MM / NEEDLE_VOXELS) ** (1 / 3)
    return min(max(NEEDLE_GRID_MM, least), WHOLE_GRID_MM)


def render_stone(
    r: StoneRecipe,
    out: Path,
    spp: int = SPP,
    optics: OpticsTable | None = None,
    close_ups: bool = True,
    hollow: Literal["air", "stand-in"] = "air",
    sparkle: bool = False,
) -> dict[str, object]:
    """One stone's views (module docstring). hollow: a crystal-lined hollow in the whole
    face's look drawn as air (its void box built here), or the stand-in; the backlit
    slice and the close-ups keep the stand-in. sparkle: also the whole face's sparkle view
    when its hollow is drawn as air (SPARKLE_LAMP)."""
    optics = optics_for(r, optics)
    t0 = time.perf_counter()
    specimen, _ = build(r)
    lo, hi = specimen.volume.solid.bounds()
    spacing = float(np.max(np.asarray(hi) - np.asarray(lo))) / WHOLE_PX
    # room round the stone for its rind, drawn on the look after the render
    pad = r.rind.reach_mm() if r.rind is not None else 0.0
    cut = cut_of(specimen, r, spacing, pad)
    record = stone_record(specimen, np.random.default_rng(r.seed))
    rng = np.random.default_rng(r.seed + 1)
    row: dict[str, object] = {"stone": r.index + 1, "summary": r.summary()}
    name = f"stone-{r.index + 1:02d}"
    grid_mm = whole_grid_mm(r, cut)
    row["whole_grid_mm"] = round(grid_mm, 3)
    # the hollow drawn as air in the whole face's look (ADR 0019 s.6.12 step 4)
    box = None
    if hollow == "air":
        tb = time.perf_counter()
        box = build_void_box(specimen.volume)
        row["box_s"] = round(time.perf_counter() - tb, 1)
    row["hollow"] = "air" if box is not None else "stand-in"
    row |= views(specimen, cut, r, record, grid_mm, optics, spp, rng, out / name, "whole",
                 box=box, sparkle=sparkle)  # fmt: skip
    Image.fromarray(preview_rgb(cut, r)).save(out / f"{name}-preview.png")
    if r.dust is not None and close_ups:
        near = window_cut(specimen, cut, dustiest(cut, CLOSE_MM), CLOSE_MM, CLOSE_PIXEL_MM)
        row |= views(specimen, near, r, record, CLOSE_GRID_MM, optics, spp, rng,
                     out / f"{name}-close", "close", CLOSE_LOOK_MM)  # fmt: skip
        Image.fromarray(preview_rgb(near, r)).save(out / f"{name}-close-preview.png")
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def views(
    specimen: Specimen,
    cut: SpecimenCut,
    recipe: StoneRecipe,
    record: BandRecord,
    grid_mm: float,
    optics: OpticsTable,
    spp: int,
    rng: np.random.Generator,
    stem: Path,
    key: str,
    look_mm: float = LOOK_DEPTH_MM,
    box: VoidBox | None = None,
    sparkle: bool = False,
) -> dict[str, object]:
    """The look (`look_mm` deep) and the backlit slice of one cut, on one grid fill;
    saves `<stem>-look.png` (host, specks and rind drawn on, `dressed_look`) and
    `<stem>-backlit.png` (as rendered). With the stone's void box the look draws the
    hollow as air (the grid continues down under it); the slice keeps the stand-in. With
    sparkle and a void box, also `<stem>-sparkle.png`: the look plus SPARKLE_LAMP on the
    same fill (`sparkle_rgb`), dressed as the look."""
    hollow: Literal["air", "stand-in"] = "air" if box is not None else "stand-in"
    look =CutRenderer(specimen, cut, Rendering.look(record, grid_mm=grid_mm, optics=optics,
                                                     depth_mm=look_mm, hollow=hollow),
                       rng, fill_depth_mm=max(SLICE_MM, look_mm), box=box)  # fmt: skip
    back = look.on(Rendering.transmitted(record, grid_mm=grid_mm, depth_mm=SLICE_MM,
                                         optics=optics, exposure=BACKLIT_EXPOSURE,
                                         rolloff_knee=BACKLIT_KNEE))  # fmt: skip
    row: dict[str, object] = {f"{key}_fill_s": round(look.fill_s, 1)}
    for view, cr in zip(VIEWS, (look, back), strict=True):
        cr.more(spp)
        rgb = dressed_look(specimen, cut, recipe, cr.rgb) if view == "look" else cr.rgb
        Image.fromarray(rgb[::-1]).save(f"{stem}-{view}.png")
        row |= {f"{key}_{view}_s": round(cr.render_s, 1), f"{key}_{view}_noise": round(cr.noise, 2)}
    row[f"{key}_look_exposure"] = round(look.exposure, 2)
    if sparkle and box is not None:
        lamp = look.on(Rendering.lamp(record, SPARKLE_LAMP, grid_mm=grid_mm, optics=optics,
                                      depth_mm=look_mm, hollow="air"))  # fmt: skip
        lamp.more(max(2, round(spp * SPARKLE_SPP_SHARE / 2) * 2))
        rgb = dressed_look(specimen, cut, recipe, sparkle_rgb(look, lamp))
        Image.fromarray(rgb[::-1]).save(f"{stem}-sparkle.png")
        row[f"{key}_sparkle_s"] = round(lamp.render_s, 1)
    return row


COLUMNS = {
    "look": "look (dome, no face glare)",
    "backlit": f"backlit {SLICE_MM:g} mm slice",
    "sparkle": "sparkle (small lamp)",
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
    """Render a variety sheet's stones."""
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
        "--stand-in", action="store_true", help="hollows as the dark stand-in, not air"
    )
    parser.add_argument(
        "--sparkle", action="store_true", help="also the sparkle view (a small lamp) of hollows"
    )
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
    recipes = draw_sheet(args.seed)
    if args.stones:
        recipes = [r for r in recipes if r.index + 1 in args.stones]
    for n, r in enumerate(recipes):
        progress.report(n / len(recipes), f"stone {r.index + 1}")
        if r.index + 1 in done:
            continue
        optics = OpticsTable(phi_iron_oxide=OpticsTable().phi_iron_oxide * args.phi_scale)
        row = render_stone(r, out, args.spp, optics, close_ups=not args.whole_only,
                           hollow="stand-in" if args.stand_in else "air",
                           sparkle=args.sparkle)  # fmt: skip
        done[r.index + 1] = row
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    rows = [done[k] for k in sorted(done)]
    print(f"saved {contact_sheet(out, rows)}")


if __name__ == "__main__":
    main()
