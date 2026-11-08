"""A preview sheet of stand-in cuts, one per stone of a drawn sheet (ADR 0018 s.1, s.7).

Screening only, for the owner's eye before any render: chalcedony takes agatesim's
stand-in colour (`chalcedony_colour`, as `nodule.descriptors.stand_in_rgb`), quartz
a glassy grey shaded per grain (tinted for amethyst or smoky centres), open space
near black, and the host a flat colour by setting. No glow, sub-banding or rind:
a render differs. Appearance only (ADR 0002); every colour here is `effective`.

Run: `uv run --no-sync python -m nodule.sheet --seed 1 --out runs/variety-sheet`.
Reports time and peak traced memory per stone (CLAUDE.md).
"""

from __future__ import annotations

import json
import math
import time
import tracemalloc
from pathlib import Path

import numpy as np
from agatesim import progress
from agatesim.fields import Phase
from agatesim.render import QUARTZ_CLEAR, QUARTZ_MILKY, chalcedony_colour
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from nodule.specimen import SpecimenCut
from nodule.variety import MAX_SLOPE, StoneRecipe, build, cut_of, draw_sheet

HOST_RGB = {  # effective: basalt grey, rhyolite pink-brown, sediment tan
    "volcanic": (68, 66, 66),
    "thunder-egg": (150, 112, 98),
    "sedimentary": (148, 128, 104),
}
VOID_RGB = (16, 16, 20)
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
    rgb[cut.inside & ((f.phase == Phase.VOID) | (f.phase == Phase.INCLUSION))] = VOID_RGB
    host = (
        "thunder-egg"
        if recipe.outline == "thunder-egg"
        else ("volcanic" if recipe.outline in ("vesicle", "prolate") else "sedimentary")
    )
    rgb[~cut.inside] = HOST_RGB[host]
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)[::-1]


def extent(cut_shape: tuple[int, int], spacing_mm: float) -> float:
    return max(cut_shape) * spacing_mm


def make_sheet(
    seed: int, n: int, px: int, out: Path, tile: int = 400, max_slope: float = MAX_SLOPE
) -> Path:
    recipes = draw_sheet(seed, n)
    out.mkdir(parents=True, exist_ok=True)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * tile, rows * (tile + 50)), (235, 235, 235))
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
        x, y = (i % cols) * tile, (i // cols) * (tile + 50)
        sheet.paste(img, (x + (tile - img.width) // 2, y + (tile - img.height) // 2))
        draw = ImageDraw.Draw(sheet)
        bar = round(10.0 / spacing * img.width / full_width)  # 10 mm scale bar
        right = x + 8 + bar
        draw.rectangle((x + 8, y + tile + 36, right, y + tile + 40), fill=(0, 0, 0))
        draw.text((right + 4, y + tile + 32), "10 mm", fill=(0, 0, 0))
        words = rec.summary().split(", ")
        draw.text((x + 6, y + tile + 2), f"{i + 1}. " + ", ".join(words[:3]), fill=(0, 0, 0))
        draw.text((x + 6, y + tile + 16), ", ".join(words[3:]), fill=(60, 60, 60))
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
    path = out / f"sheet-seed{seed}{tag}.png"
    sheet.save(path)
    (out / f"sheet-seed{seed}{tag}.json").write_text(json.dumps(log, indent=1), "utf-8")
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


def slope_pairs(seed: int, caps: tuple[float, float], px: int, out: Path, tile: int = 360) -> Path:
    """Every warped stone of a sheet at two no-fold caps, side by side: whole stone
    and a 15 mm close-up at each cap (ADR 0018 s.8). Mild-warp stones are left out
    (their wobble is under both clamps, so they do not change)."""
    recipes = [r for r in draw_sheet(seed) if r.warp != "mild"]
    out.mkdir(parents=True, exist_ok=True)
    head = 24
    sheet = Image.new("RGB", (4 * tile, head + len(recipes) * (tile + 22)), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    for k, cap in enumerate(caps):
        draw.text(
            (2 * k * tile + 6, 6), f"cap {cap:g}: whole stone | 15 mm close-up", fill=(0, 0, 0)
        )
    for row, rec in enumerate(recipes):
        y = head + row * (tile + 22)
        for k, cap in enumerate(caps):
            specimen, _ = build(rec, cap)
            lo, hi = specimen.volume.solid.bounds()
            spacing = float(np.max(np.asarray(hi) - np.asarray(lo))) / px
            cut = cut_of(specimen, rec, spacing)
            rgb = preview_rgb(cut, rec)
            whole = Image.fromarray(rgb)
            whole.thumbnail((tile, tile))
            near = close_up(cut, rgb, 15.0).resize((tile, tile), Image.Resampling.NEAREST)
            x = 2 * k * tile
            sheet.paste(whole, (x + (tile - whole.width) // 2, y + (tile - whole.height) // 2))
            sheet.paste(near, (x + tile, y))
        draw.text((6, y + tile + 4), f"stone {rec.index + 1}. {rec.summary()}", fill=(0, 0, 0))
        print(f"stone {rec.index + 1} done ({rec.warp})")
    path = out / f"slope-pairs-seed{seed}.png"
    sheet.save(path)
    return path


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
        "--pairs",
        type=float,
        nargs=2,
        metavar=("CAP_A", "CAP_B"),
        help="instead of a sheet, show the warped stones at two caps side by side",
    )
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    if args.pairs:
        path = slope_pairs(args.seed, tuple(args.pairs), args.px, args.out)
    else:
        path = make_sheet(args.seed, args.n, args.px, args.out, max_slope=args.max_slope)
    print(f"saved {path} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
