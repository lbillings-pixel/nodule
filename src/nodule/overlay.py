"""Outline-check pictures for the labelling batches (ADR 0004, section 8 and
choice 9b).

C1-C3 count only once the outline has been seen to follow the stone's rim. For
each usable library photo this draws, at agatesim's working size:

- the outline agatesim's segmentation found (green);
- its convex hull (cyan), against which concavities are measured;
- each counted concavity (C3) as a red line from its floor to the hull, along
  the ray from the inscribed circle's centre (yellow dot);
- a note in the corner: "touches frame" where the outline runs off the photo
  (C1-C3 unknown without a check), and the concavity count.

Pictures go to `runs/outline-check/` (gitignored), named by agatesim library
id, as plain PNG with no metadata. They are made from the library's Commons
photos, never from the owner's specimens.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from agatesim import _scipy as agatesim_scipy  # pyright: ignore[reportPrivateUsage]
from agatesim import progress
from agatesim.library import load_labels, load_library
from agatesim.shapes import WORKING_PX
from agatesim.specimen.photo import load_photo
from agatesim.specimen.segment import segment
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from nodule import outline
from nodule._scipy import hull_2d
from nodule.tags import ROOT, agatesim_dir, image_path

OUT_DIR = ROOT / "runs" / "outline-check"
GREEN, CYAN, RED, YELLOW = (40, 230, 60), (0, 220, 255), (255, 40, 40), (255, 230, 0)


def edge(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Pixels of the mask with a neighbour outside it."""
    inner = mask.copy()
    inner[1:] &= mask[:-1]
    inner[:-1] &= mask[1:]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return mask & ~inner


def caption(region: NDArray[np.bool_], found: int) -> str:
    """The corner note for a filled region with `found` counted concavities."""
    if not region.any():
        return "no outline found"
    note = f"{found} concavities"
    if outline.touches_frame(region):
        note += "; touches frame: C1-C3 unknown"
    return note


def label(img: Image.Image, note: str) -> Image.Image:
    pen = ImageDraw.Draw(img)
    pen.rectangle([0, 0, 8 * len(note) + 16, 24], fill=(0, 0, 0))
    pen.text((8, 6), note, fill=(255, 255, 255))
    return img


def draw(rgb: NDArray[np.uint8], mask: NDArray[np.bool_]) -> Image.Image:
    """The photo with its outline, hull and counted concavities drawn on."""
    img = Image.fromarray(rgb).convert("RGB")
    region = outline.filled(mask)
    width = max(2, round(max(mask.shape) / 600))
    if not region.any():
        return label(img, caption(region, 0))
    pixels = np.asarray(img).copy()
    line = edge(region)
    pixels[agatesim_scipy.dilate(line, width - 1) if width > 1 else line] = GREEN
    img = Image.fromarray(pixels)
    pen = ImageDraw.Draw(img)
    rows, cols = np.nonzero(region)
    vertices, _ = hull_2d(np.stack([cols, rows], axis=1).astype(np.float64))
    pen.line([(float(x), float(y)) for x, y in vertices] + [tuple(vertices[0])], CYAN, width)
    centre, _ = outline.inscribed_circle(region)
    found = outline.concavities(region, centre)
    cx, cy = float(centre[0]), float(centre[1])
    for c in found:
        dx, dy = c.direction
        pen.line([(cx + c.x * dx, cy + c.x * dy), (cx + c.y * dx, cy + c.y * dy)], RED, width)
    r = 3 * width
    pen.ellipse([cx - r, cy - r, cx + r, cy + r], fill=YELLOW)
    return label(img, caption(region, len(found)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="draw outline-check pictures (ADR 0004)")
    parser.add_argument("--sample", type=int, help="draw only this many photos, chosen at random")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="folder for the pictures")
    args = parser.parse_args(argv)
    agatesim = agatesim_dir()
    refs = agatesim / "references"
    if not (refs / "library.toml").exists():
        print(f"no agatesim library at {agatesim}; set NODULE_AGATESIM")
        return 1
    library = load_library(refs / "library.toml")
    labels = load_labels(refs / "library-labels.toml")
    ids = sorted(k for k, v in labels.items() if v.verdict != "unusable")
    if args.sample is not None:
        ids = sorted(np.random.default_rng(0).choice(ids, size=args.sample, replace=False).tolist())
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for n, image_id in enumerate(ids):
        progress.report(n / max(len(ids), 1), "drawing outlines")
        path = image_path(agatesim, library.get(image_id))
        if not path.exists():
            continue
        rgb = load_photo(path.read_bytes(), WORKING_PX).rgb
        draw(rgb, segment(rgb).mask).save(args.out / f"{image_id}.png")
    print(f"drew {len(ids)} photos into {args.out} in {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
