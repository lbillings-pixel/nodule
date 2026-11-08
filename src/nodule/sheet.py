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
from nodule.variety import StoneRecipe, build, cut_of, draw_sheet

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


def make_sheet(seed: int, n: int, px: int, out: Path, tile: int = 400) -> Path:
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
        specimen, _ = build(rec)
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
    path = out / f"sheet-seed{seed}.png"
    sheet.save(path)
    (out / f"sheet-seed{seed}.json").write_text(json.dumps(log, indent=1), "utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    """Draw a sheet of stones, cut each, and save a stand-in preview sheet."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--px", type=int, default=600, help="pixels across a stone's box")
    parser.add_argument("--out", type=Path, default=Path("runs/variety-sheet"))
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    path = make_sheet(args.seed, args.n, args.px, args.out)
    print(f"saved {path} in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
