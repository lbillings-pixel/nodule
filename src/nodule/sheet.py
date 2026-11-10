"""A preview sheet of stand-in cuts, one per stone of a drawn sheet (ADR 0018 s.1, s.7).

Screening only, for the owner's eye before any render: chalcedony takes agatesim's
stand-in colour (`chalcedony_colour`, as `nodule.descriptors.stand_in_rgb`), quartz
a glassy grey shaded per grain (tinted for amethyst or smoky centres), open space
near black, plumes by their tint (`nodule.plumes`), and the host a flat colour by
setting. No glow, sub-banding or rind: a render differs. Appearance only (ADR
0002); every colour here is `effective`.

Run: `uv run --no-sync python -m nodule.sheet --seed 1 --out runs/variety-sheet`.
Reports time and peak traced memory per stone (CLAUDE.md).
"""

from __future__ import annotations

import json
import math
import textwrap
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path

import numpy as np
from agatesim import progress
from agatesim.fields import Phase
from agatesim.render import QUARTZ_CLEAR, QUARTZ_MILKY, chalcedony_colour
from agatesim.volume.slicing import reshape_fields
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from nodule.dust import dust_excess
from nodule.plumes import inclusion_rgb
from nodule.specimen import Specimen, SpecimenCut
from nodule.variety import MAX_SLOPE, Roughness, StoneRecipe, build, cut_of, draw_sheet

HOST_RGB = {  # effective: basalt grey, rhyolite pink-brown, sediment tan
    "volcanic": (68, 66, 66),
    "thunder-egg": (150, 112, 98),
    "sedimentary": (148, 128, 104),
}
VOID_RGB = (16, 16, 20)
LABEL_H = 64  # label rows under each stone: summary on up to 3 lines, scale bar
LABEL_CHARS = 64  # characters per label line in a 400 px tile
TINT = {"amethyst": np.array([1.0, 0.82, 1.08]), "smoky": np.array([0.72, 0.68, 0.64])}


def preview_rgb(cut: SpecimenCut, recipe: StoneRecipe) -> NDArray[np.uint8]:
    """The stand-in picture of a cut, rows top to bottom (+v up)."""
    f = cut.fields
    trans = np.nan_to_num(f.translucency, nan=0.0).astype(np.float64)
    iron = np.nan_to_num(f.impurity, nan=0.0).astype(np.float64)
    rgb = chalcedony_colour(trans, iron, f.pigment)
    quartz = cut.inside & (f.phase == Phase.QUARTZ)
    if quartz.any():
        shade = ((f.grain[quartz].astype(np.int64) * 7919 + 13) % 1000) / 1000.0
        q = (0.45 + 0.35 * shade[:, None]) * QUARTZ_CLEAR + (
            0.55 - 0.35 * shade[:, None]
        ) * QUARTZ_MILKY
        rgb[quartz] = q * TINT.get(recipe.tint, 1.0)
    rgb[cut.inside & (f.phase == Phase.VOID)] = VOID_RGB
    # Plumes (and tubes) by their tint through nodule's ramp (`nodule.plumes`).
    inclusion = cut.inside & (f.phase == Phase.INCLUSION)
    rgb[inclusion] = inclusion_rgb(f.impurity[inclusion])
    host = (
        "thunder-egg"
        if recipe.outline == "thunder-egg"
        else ("volcanic" if recipe.outline in ("vesicle", "prolate") else "sedimentary")
    )
    rgb[~cut.inside] = HOST_RGB[host]
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)[::-1]


def extent(cut_shape: tuple[int, int], spacing_mm: float) -> float:
    return max(cut_shape) * spacing_mm


def plume_recipes(seed: int, n: int) -> list[StoneRecipe]:
    """The first n plume stones of the sheets from `seed` on (ADR 0018 s.13), for a
    look sheet of plumes alone."""
    found: list[StoneRecipe] = []
    while len(found) < n:
        found += [r for r in draw_sheet(seed) if r.plumes is not None]
        seed += 1
    return found[:n]


def make_sheet(
    seed: int,
    n: int,
    px: int,
    out: Path,
    tile: int = 400,
    max_slope: float = MAX_SLOPE,
    recipes: list[StoneRecipe] | None = None,
    name: str | None = None,
) -> Path:
    recipes = draw_sheet(seed, n) if recipes is None else recipes
    n = len(recipes)
    out.mkdir(parents=True, exist_ok=True)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * tile, rows * (tile + LABEL_H)), (235, 235, 235))
    log: list[dict[str, object]] = []
    for i, rec in enumerate(recipes):
        progress.report(i / n, f"stone {i + 1} of {n}")
        tracemalloc.start()
        t0 = time.perf_counter()
        specimen, _ = build(rec, max_slope)
        t1 = time.perf_counter()
        lo, hi = specimen.volume.solid.bounds()
        spacing = float(np.max(np.asarray(hi) - np.asarray(lo))) / px
        cut = cut_of(specimen, rec, spacing)
        img = Image.fromarray(preview_rgb(cut, rec))
        t2 = time.perf_counter()
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        img.save(out / f"stone-{i + 1:02d}.png")
        full_width = img.width
        img.thumbnail((tile, tile))
        x, y = (i % cols) * tile, (i // cols) * (tile + LABEL_H)
        sheet.paste(img, (x + (tile - img.width) // 2, y + (tile - img.height) // 2))
        draw = ImageDraw.Draw(sheet)
        bar = round(10.0 / spacing * img.width / full_width)  # 10 mm scale bar
        right = x + 8 + bar
        draw.rectangle((x + 8, y + tile + 50, right, y + tile + 54), fill=(0, 0, 0))
        draw.text((right + 4, y + tile + 46), "10 mm", fill=(0, 0, 0))
        words = rec.summary().split(", ")
        draw.text((x + 6, y + tile + 2), f"{i + 1}. " + ", ".join(words[:3]), fill=(0, 0, 0))
        rest = textwrap.wrap(", ".join(words[3:]), LABEL_CHARS)[:2]
        for k, line in enumerate(rest):
            draw.text((x + 6, y + tile + 16 + 14 * k), line, fill=(60, 60, 60))
        size_mm = float(np.max(np.asarray(hi) - np.asarray(lo)))
        print(
            f"stone {i + 1}: build {t1 - t0:.1f} s, cut+colour {t2 - t1:.1f} s, "
            f"peak {peak:.0f} MB, {size_mm:.0f} mm; {rec.summary()}"
        )
        log.append(
            {
                "recipe": rec.model_dump(mode="json"),
                "build_s": round(t1 - t0, 2),
                "cut_s": round(t2 - t1, 2),
                "peak_mb": round(peak),
            }
        )
    progress.report(1.0, "sheet done")
    tag = "" if max_slope == MAX_SLOPE else f"-cap{max_slope:g}"
    path = out / (name or f"sheet-seed{seed}{tag}.png")
    sheet.save(path)
    path.with_suffix(".json").write_text(json.dumps(log, indent=1), "utf-8")
    return path


def close_up(cut: SpecimenCut, rgb: NDArray[np.uint8], window_mm: float) -> Image.Image:
    """A square window of the picture, centred halfway from the stone's centre to
    its right-hand rim on the centre row: where wall bands sit in most stones."""
    inside = cut.inside[::-1]
    rows, cols = np.nonzero(inside)
    r0 = int(np.median(rows))
    c_mid = int(np.median(cols))
    right = int(cols[rows == r0].max())
    c0 = (c_mid + right) // 2
    half = round(window_mm / 2 / cut.spacing_mm)
    box = (c0 - half, r0 - half, c0 + half, r0 + half)
    return Image.fromarray(rgb).crop(box)


Variant = tuple[str, float, "Roughness | None"]  # label, no-fold cap, band roughness
# The owner's roughness trial (ADR 0018 s.8) chose smooth roughness, weaker than
# the 0.4 mm tried (s.9): plain, then the faint and marked ends of the new range.
ROUGHNESS_VARIANTS: list[Variant] = [
    ("plain (cap 0.5)", MAX_SLOPE, None),
    ("faint roughness", MAX_SLOPE, Roughness(amplitude_mm=0.08, scale_mm=2.0, ridged=False)),
    ("marked roughness", MAX_SLOPE, Roughness(amplitude_mm=0.2, scale_mm=2.0, ridged=False)),
]


def variant_sheet(
    recipes: list[StoneRecipe], variants: list[Variant], px: int, path: Path, tile: int = 330
) -> Path:
    """Each stone in each variant side by side: whole stone and a 15 mm close-up
    per variant (ADR 0018 s.8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    head, cols = 24, 2 * len(variants)
    sheet = Image.new("RGB", (cols * tile, head + len(recipes) * (tile + 22)), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    for k, (label, _, _) in enumerate(variants):
        draw.text((2 * k * tile + 6, 6), f"{label}: whole | 15 mm close-up", fill=(0, 0, 0))
    for row, rec in enumerate(recipes):
        y = head + row * (tile + 22)
        for k, (_, cap, roughness) in enumerate(variants):
            stone = rec.model_copy(update={"roughness": roughness})
            specimen, _ = build(stone, cap)
            lo, hi = specimen.volume.solid.bounds()
            spacing = float(np.max(np.asarray(hi) - np.asarray(lo))) / px
            cut = cut_of(specimen, stone, spacing)
            rgb = preview_rgb(cut, stone)
            whole = Image.fromarray(rgb)
            whole.thumbnail((tile, tile))
            near = close_up(cut, rgb, 15.0).resize((tile, tile), Image.Resampling.NEAREST)
            x = 2 * k * tile
            sheet.paste(whole, (x + (tile - whole.width) // 2, y + (tile - whole.height) // 2))
            sheet.paste(near, (x + tile, y))
        draw.text((6, y + tile + 4), f"stone {rec.index + 1}. {rec.summary()}", fill=(0, 0, 0))
        print(f"stone {rec.index + 1} done")
    sheet.save(path)
    return path


def window_cut(
    specimen: Specimen,
    cut: SpecimenCut,
    centre: tuple[float, float],
    size_mm: float,
    spacing_mm: float,
) -> SpecimenCut:
    """A square window of a cut's plane at its own spacing, pixel centres on integer
    multiples of it (as `Specimen.cut`), so dust thinner than a whole-stone pixel shows."""
    axes: list[NDArray[np.float64]] = []
    for c in centre:
        k0 = math.floor((c - size_mm / 2) / spacing_mm)
        axes.append(
            (np.arange(k0, k0 + round(size_mm / spacing_mm)) * spacing_mm).astype(np.float64)
        )
    s, t = axes
    ss, tt = np.meshgrid(s, t)
    fields, inside = specimen.evaluate(cut.plane.points(ss, tt), spacing_mm)
    return SpecimenCut(
        plane=cut.plane,
        spacing_mm=spacing_mm,
        s=s,
        t=t,
        fields=reshape_fields(fields, ss.shape),
        inside=inside.reshape(ss.shape),
        fracture_mm=np.full(ss.shape, np.inf),
    )


def dustiest(cut: SpecimenCut, size_mm: float) -> tuple[float, float]:
    """Centre (s, t) of the window of a cut that holds the most dust: the place to
    show the owner, not a typical one."""
    f = cut.fields
    banded = cut.inside & (f.phase == Phase.CHALCEDONY) & (f.band >= 0)
    ex = dust_excess(f.impurity, f.band, banded)
    k = max(1, round(size_mm / cut.spacing_mm))
    c = np.cumsum(np.cumsum(np.pad(ex, ((1, 0), (1, 0))), 0), 1)
    box = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    j, i = np.unravel_index(int(np.argmax(box)), box.shape)
    return float(cut.s[i + k // 2]), float(cut.t[j + k // 2])


def dust_sheet(seed: int, px: int, out: Path, tile: int = 300) -> Path:
    """Each dusty stone of a sheet in a row (ADR 0018 s.11): whole stone with dust;
    a 12 mm window without and with dust (0.03 mm pixels); a 4 mm window with dust
    (0.01 mm pixels). Windows sit where the stone's dust is densest."""
    recipes = [r for r in draw_sheet(seed) if r.dust is not None]
    out.mkdir(parents=True, exist_ok=True)
    head = 24
    sheet = Image.new("RGB", (4 * tile, head + len(recipes) * (tile + 22)), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    labels = ["whole stone", "12 mm, no dust", "12 mm, dust", "4 mm, dust"]
    for k, label in enumerate(labels):
        draw.text((k * tile + 6, 6), label, fill=(0, 0, 0))
    for row, rec in enumerate(recipes):
        t0 = time.perf_counter()
        specimen, layers = build(rec)
        lo, hi = specimen.volume.solid.bounds()
        spacing = float(np.max(np.asarray(hi) - np.asarray(lo))) / px
        cut = cut_of(specimen, rec, spacing)
        plain = replace(specimen, layers=replace(layers, dust=None))
        wide = dustiest(cut, 12.0)
        whole = Image.fromarray(preview_rgb(cut, rec))
        whole.thumbnail((tile, tile))
        wide_cut = window_cut(specimen, cut, wide, 12.0, 0.03)
        near = dustiest(wide_cut, 4.0)
        tiles = [
            whole,
            Image.fromarray(preview_rgb(window_cut(plain, cut, wide, 12.0, 0.03), rec)),
            Image.fromarray(preview_rgb(wide_cut, rec)),
            Image.fromarray(preview_rgb(window_cut(specimen, cut, near, 4.0, 0.01), rec)),
        ]
        y = head + row * (tile + 22)
        for k, im in enumerate(tiles):
            if k:
                im = im.resize((tile, tile), Image.Resampling.LANCZOS)
            sheet.paste(im, (k * tile + (tile - im.width) // 2, y + (tile - im.height) // 2))
        draw.text((6, y + tile + 4), f"stone {rec.index + 1}. {rec.summary()}", fill=(0, 0, 0))
        print(f"stone {rec.index + 1}: {time.perf_counter() - t0:.1f} s; {rec.summary()}")
    path = out / f"dust-sheet-seed{seed}.png"
    sheet.save(path)
    return path


def slope_pairs(seed: int, caps: tuple[float, float], px: int, out: Path) -> Path:
    """Every warped stone of a sheet at two no-fold caps (mild-warp stones are
    under both clamps, so they do not change)."""
    recipes = [r for r in draw_sheet(seed) if r.warp != "mild"]
    variants: list[Variant] = [(f"cap {c:g}", c, None) for c in caps]
    return variant_sheet(recipes, variants, px, out / f"slope-pairs-seed{seed}.png", tile=360)


def roughness_trial(seeds: list[int], px: int, out: Path) -> Path:
    """The crenulated stones of the given seeds, plain and with band roughness."""
    recipes = [r for s in seeds for r in draw_sheet(s) if r.warp == "crenulated"]
    name = "roughness-trial-seeds" + "-".join(str(s) for s in seeds) + ".png"
    return variant_sheet(recipes, ROUGHNESS_VARIANTS, px, out / name)


def main(argv: list[str] | None = None) -> None:
    """Draw a sheet of stones, cut each, and save a stand-in preview sheet."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--px", type=int, default=600, help="pixels across a stone's box")
    parser.add_argument("--out", type=Path, default=Path("runs/variety-sheet"))
    parser.add_argument("--max-slope", type=float, default=MAX_SLOPE, help="no-fold cap")
    parser.add_argument(
        "--roughness-trial",
        type=int,
        nargs="+",
        metavar="SEED",
        help="instead of a sheet, the crenulated stones of these seeds with and without roughness",
    )
    parser.add_argument("--dust", action="store_true", help="instead, the dust look sheet")
    parser.add_argument(
        "--plumes", action="store_true", help="instead, n plume stones from sheets seed on"
    )
    parser.add_argument(
        "--pairs",
        type=float,
        nargs=2,
        metavar=("CAP_A", "CAP_B"),
        help="instead of a sheet, show the warped stones at two caps side by side",
    )
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    if args.roughness_trial:
        path = roughness_trial(args.roughness_trial, args.px, args.out)
    elif args.plumes:
        recipes = plume_recipes(args.seed, args.n)
        name = f"sheet-plumes-seed{args.seed}.png"
        path = make_sheet(args.seed, args.n, args.px, args.out, recipes=recipes, name=name)
    elif args.dust:
        path = dust_sheet(args.seed, args.px, args.out)
    elif args.pairs:
        path = slope_pairs(args.seed, tuple(args.pairs), args.px, args.out)
    else:
        path = make_sheet(args.seed, args.n, args.px, args.out, max_slope=args.max_slope)
    print(f"saved {path} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
