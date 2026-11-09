"""Renders of a variety sheet's stones with their layers (ADR 0019): the first look at
ADR 0018's features through the path tracer instead of the stand-in colours.

Per stone the renderer can show (no open cavity yet: ADR 0019 option for the owner):
the whole face under the dome light (L1, ADR 0008 7c; black bottom, ADR 0014) beside its
stand-in preview, and for a dusty stone a 12 mm close-up at 0.03 mm pixels where its
dust is densest. No photo blur or noise: these are for the eye, not the blind test.

Run: `uv run --no-sync python -m nodule.varietyrender --seed 1 --out runs/variety-render`.
GPU. Logs time per stone; resumes, skipping stones already saved.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from agatesim import progress
from PIL import Image, ImageDraw

from nodule.render.cutface import CutRenderer, Rendering
from nodule.render.fakes import stone_record
from nodule.render.materials import OpticsTable
from nodule.sheet import dustiest, preview_rgb, window_cut
from nodule.variety import StoneRecipe, build, cut_of, draw_sheet

WHOLE_PX = 600  # pixels across the stone's box
WHOLE_GRID_MM = 0.1
CLOSE_MM, CLOSE_PIXEL_MM, CLOSE_GRID_MM = 12.0, 0.03, 0.03
SPP = 512  # samples per pixel (effective; noise falls as 1/sqrt(spp))


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
    optics = optics or OpticsTable()
    t0 = time.perf_counter()
    specimen, _ = build(r)
    lo, hi = specimen.volume.solid.bounds()
    spacing = float(np.max(np.asarray(hi) - np.asarray(lo))) / WHOLE_PX
    cut = cut_of(specimen, r, spacing)
    record = stone_record(specimen, np.random.default_rng(r.seed))
    rng = np.random.default_rng(r.seed + 1)
    row: dict[str, object] = {"stone": r.index + 1, "summary": r.summary()}
    whole = CutRenderer(
        specimen, cut, Rendering.reflected(record, grid_mm=WHOLE_GRID_MM, optics=optics), rng
    )
    whole.more(spp)
    Image.fromarray(whole.rgb[::-1]).save(out / f"stone-{r.index + 1:02d}-render.png")
    Image.fromarray(preview_rgb(cut, r)).save(out / f"stone-{r.index + 1:02d}-preview.png")
    row |= {"whole_fill_s": round(whole.fill_s, 1), "whole_render_s": round(whole.render_s, 1),
            "noise": round(whole.noise, 2)}  # fmt: skip
    if r.dust is not None and close_ups:
        near = window_cut(specimen, cut, dustiest(cut, CLOSE_MM), CLOSE_MM, CLOSE_PIXEL_MM)
        close = CutRenderer(
            specimen, near, Rendering.reflected(record, grid_mm=CLOSE_GRID_MM, optics=optics), rng
        )
        close.more(spp)
        Image.fromarray(close.rgb[::-1]).save(out / f"stone-{r.index + 1:02d}-close-render.png")
        Image.fromarray(preview_rgb(near, r)).save(
            out / f"stone-{r.index + 1:02d}-close-preview.png"
        )
        row |= {"close_fill_s": round(close.fill_s, 1), "close_render_s": round(close.render_s, 1)}
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def contact_sheet(out: Path, rows: list[dict[str, object]], tile: int = 360) -> Path:
    """Render | preview, then close-up render | preview where there is one."""
    head = 24
    sheet = Image.new("RGB", (4 * tile, head + len(rows) * (tile + 22)), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    for k, label in enumerate(["render", "stand-in preview", "12 mm render", "12 mm preview"]):
        draw.text((k * tile + 6, 6), label, fill=(0, 0, 0))
    for n, row in enumerate(rows):
        y = head + n * (tile + 22)
        i = int(str(row["stone"]))
        names = ["render", "preview", "close-render", "close-preview"]
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
        "--phi-scale", type=float, default=1.0, help="iron-oxide loading x this (ADR 0019 B)"
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
