"""The owner's check of round 2's drawn lines (ADR 0006, section 9, choice 7o).

`python -m nodule.linecheck --draft N` drafts the line of the first N faces of
round 2's pool (`zones.round2_pool`) with `nodule.zoneline`. The face is the
rim outline where it was judged "yes", else agatesim's working region (as
`nodule.tags`); the centre comes from the band paths where agatesim's cache
holds the band spacing, else from the face's deepest point. Into
runs/zone-lines/ (never committed) go each photo at 1 MP as plain PNG with no
metadata, its face (to move the line) and the drafts.

`python -m nodule.linecheck` then serves a page on this PC only (127.0.0.1),
like `nodule.zonecheck`, in the pool's order. It shows each line as the
helpers will see it (`ticked`: ticks every 0.1, 0 at the rim, 1 at the
centre). The owner accepts it, clicks the centre where the draft is off,
takes the next direction for a stated reason, or marks "no banded face" (the
pool's next then stands in). The server, not the page, works out the rim;
saves are checked (`zones.line_problems`) and written beside the labels in
references/band-zone-labels.toml. Done when `zones.BLIND` faces have a line.

`--ticked` draws the checked lines of round 2's sample into
runs/zone-lines/ticked/ for the helpers. Nothing leaves the PC.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import threading
import time
import tracemalloc
import webbrowser
from collections.abc import Mapping, Sequence
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import numpy as np
from agatesim import progress
from agatesim.library import load_library
from agatesim.shapes import WORKING_PX
from agatesim.specimen.photo import load_photo
from agatesim.specimen.quality import luminance
from agatesim.specimen.segment import segment
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from nodule import bands, labels, rims, zoneline, zones
from nodule.check import HOST, handler
from nodule.tags import ROOT, agatesim_dir, image_path, passes

PORT = 8767
OUT_DIR = ROOT / "runs" / "zone-lines"
DRAFTS = "drafts.json"
DRAFT = 20  # faces drafted by default: the blind 10 and room for faces marked not banded
PICTURE_PX = 10**6  # the pictures' size, as for `bands`: lines are fractions, so any size holds
YELLOW, BLACK = (255, 230, 0), (0, 0, 0)


def draft_face(
    finder: rims.Finder,
    verdicts: rims.Verdicts,
    image_id: str,
    data: bytes,
    entry: dict[str, object] | None,
) -> tuple[NDArray[np.uint8], NDArray[np.bool_], float, dict[str, Any]]:
    """The photo at `PICTURE_PX`, its face, full px per face px, and the
    drafted line (with its support, where the band paths were traced)."""
    _, rim = finder.find(image_id, data)
    judged = rims.verdict(verdicts, image_id, rim.mask) if rim.found else None
    yes = judged is not None and judged.passed
    mask = rim.mask if yes else segment(load_photo(data, WORKING_PX).rgb).mask
    rgb = load_photo(data, PICTURE_PX).rgb
    shape = (rgb.shape[0], rgb.shape[1])
    spacing = entry.get("spacing_px") if entry is not None else None
    found: dict[str, Any] = {"outline": "rim" if yes else "working region"}
    if passes(entry, len(data)) and isinstance(spacing, int | float) and mask.any():
        traced = bands.trace(luminance(rgb), mask, mask.shape[1] / shape[1], float(spacing))
        line, d = zoneline.draw_line(traced, float(spacing), image_id, shape)
        face, k = traced.face, float(traced.k)
        found |= {"support": d.support, "paths": d.paths}
    else:
        line = zoneline.outline_line(mask, image_id, shape)
        face, k = mask, shape[1] / mask.shape[1]
    found |= {"rim": list(line.rim), "centre": list(line.centre), "source": line.source}
    return rgb, face, k, found


def draft(ids: Sequence[str], out: Path = OUT_DIR) -> dict[str, dict[str, Any]]:
    """Draft and store the lines of `ids` (module docstring)."""
    agatesim = agatesim_dir()
    refs = agatesim / "references"
    library = load_library(refs / "library.toml")
    cache = json.loads((refs / "models" / "shapes.json").read_text("utf-8"))
    finder, verdicts = rims.Finder(agatesim), rims.load()
    out.mkdir(parents=True, exist_ok=True)
    drafts: dict[str, dict[str, Any]] = {}
    for n, image_id in enumerate(ids):
        progress.report(n / max(len(ids), 1), "drafting lines")
        data = image_path(agatesim, library.get(image_id)).read_bytes()
        rgb, face, k, found = draft_face(finder, verdicts, image_id, data, cache.get(image_id))
        Image.fromarray(rgb).save(out / f"{image_id}.png")  # plain PNG: no metadata
        np.savez_compressed(out / f"{image_id}.npz", face=face, k=k, shape=rgb.shape[:2])
        drafts[image_id] = found
    (out / DRAFTS).write_text(json.dumps(drafts, indent=1), "utf-8")
    return drafts


def ticked(img: Image.Image, rim: Sequence[float], centre: Sequence[float]) -> Image.Image:
    """The picture with the line drawn on: ticks every 0.1 (longer at 0.5),
    0 at the rim and 1 at the centre."""
    img = img.convert("RGB")
    w, h = img.size
    (x0, y0), (x1, y1) = (rim[0] * w, rim[1] * h), (centre[0] * w, centre[1] * h)
    length = math.hypot(x1 - x0, y1 - y0) or 1.0
    nx, ny = -(y1 - y0) / length, (x1 - x0) / length  # unit normal to the line
    width = max(2, round(math.hypot(w, h) / 600))
    tick = 4 * width
    font = ImageFont.load_default(size=8 * width)
    pen = ImageDraw.Draw(img)
    marks: list[tuple[tuple[float, float], tuple[float, float]]] = [((x0, y0), (x1, y1))]
    for i in range(zoneline.TICKS + 1):
        t = i / zoneline.TICKS
        x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        s = 2 * tick if i in (0, zoneline.TICKS // 2, zoneline.TICKS) else tick
        marks.append(((x - s * nx, y - s * ny), (x + s * nx, y + s * ny)))
    for colour, extra in ((BLACK, 2 * width), (YELLOW, 0)):  # dark edge, then the line
        for a, b in marks:
            pen.line([a, b], colour, width + extra)
    for text, (x, y) in (("0", (x0, y0)), ("1", (x1, y1)), ("0.5", ((x0 + x1) / 2, (y0 + y1) / 2))):
        at = (x + 3 * tick * nx, y + 3 * tick * ny)
        pen.text(at, text, YELLOW, font, "mm", stroke_width=width, stroke_fill=BLACK)
    return img


class LineChecker:
    """Round 2's pool, its drafted lines and the C23-C25 labels file."""

    page = Path(__file__).with_name("linecheck.html")

    def __init__(
        self,
        path: Path = zones.ZONES_TOML,
        pictures: Path = OUT_DIR,
        view_labels: Path = labels.LABELS_TOML,
    ) -> None:
        self.path = path
        self.pictures = pictures
        self.images = zones.load(path)
        self.views = labels.load(view_labels)
        drafts = pictures / DRAFTS
        self.drafts: dict[str, dict[str, Any]] = (
            json.loads(drafts.read_text("utf-8")) if drafts.exists() else {}
        )
        self.sample = [k for k in zones.round2_pool(self.views, self.images) if k in self.drafts]
        self.faces: dict[str, tuple[NDArray[np.bool_], float, tuple[int, int]]] = {}
        self.lock = threading.Lock()

    def data(self) -> dict[str, object]:
        return {
            "rows": [
                {"id": k, "draft": self.drafts[k], "line": self.images.get(k, {}).get("line")}
                for k in self.sample
            ],
            "reasons": zones.LINE_REASONS,
            "needed": zones.BLIND,
        }

    def face(self, image_id: str) -> tuple[NDArray[np.bool_], float, tuple[int, int]]:
        if image_id not in self.faces:
            with np.load(self.pictures / f"{image_id}.npz") as f:
                shape = (int(f["shape"][0]), int(f["shape"][1]))
                self.faces[image_id] = (f["face"], float(f["k"]), shape)
        return self.faces[image_id]

    def line(self, image_id: str, centre: object, draw: object) -> tuple[dict[str, Any], list[str]]:
        """The line from `centre` ([x, y] fractions; None: the draft's) at the
        `draw`th direction, and what is wrong with the request."""
        if image_id not in self.sample:
            return {}, [f"{image_id!r} is not drafted in round 2's pool"]
        if type(draw) is not int or draw < 0:
            return {}, [f"draw: {draw!r} is not a whole number from 0"]
        face, k, (h, w) = self.face(image_id)
        source = "owner"
        if centre is None:
            centre, source = self.drafts[image_id]["centre"], self.drafts[image_id]["source"]
        xy = cast("list[object]", centre) if isinstance(centre, list) else []
        if len(xy) != 2 or not all(type(v) in (int, float) for v in xy):
            return {}, [f"centre: {centre!r} is not [x, y]"]
        x, y = (float(cast("float", v)) for v in xy)
        px = (y * h - 0.5, x * w - 0.5)
        if not zoneline.inside(face, k, np.array(px)):
            return {}, ["the centre must lie inside the stone's outline"]
        rim = zoneline.rim_point(face, k, px, zoneline.direction(image_id, draw))
        rim_frac = [round((rim[1] + 0.5) / w, 4), round((rim[0] + 0.5) / h, 4)]
        line = {"rim": rim_frac, "centre": [round(x, 4), round(y, 4)]}
        return line | {"source": source, "draw": draw}, []

    def render(self, name: str, query: Mapping[str, list[str]]) -> bytes | None:
        """The picture with the line asked for (`?draw=n&x=..&y=..`; no x, y:
        the draft's centre), as PNG; None if there is no such picture."""
        if name not in self.sample:
            return None
        try:
            draw = int(query.get("draw", ["0"])[0])
            x, y = query.get("x"), query.get("y")
            centre = [float(x[0]), float(y[0])] if x and y else None
        except ValueError:
            return None
        line, found = self.line(name, centre, draw)
        with Image.open(self.pictures / f"{name}.png") as img:
            out = ticked(img, line["rim"], line["centre"]) if not found else img.convert("RGB")
        buf = io.BytesIO()
        out.save(buf, "PNG")
        return buf.getvalue()

    def picture(self, name: str) -> Path | None:
        return None  # pictures come from `render`

    def save(self, image_id: str, values: dict[str, Any]) -> list[str]:
        """Write the owner's line for one face; what was wrong, if refused."""
        note = values.get("note", "")
        if values.get("banded") == "no":
            line: dict[str, Any] = {"banded": "no"}
            if image_id not in self.sample:
                return [f"{image_id!r} is not drafted in round 2's pool"]
        else:
            reasons = values.get("reasons", [])
            line, found = self.line(image_id, values.get("centre"), values.get("draw", 0))
            if found:
                return found
            if reasons:
                line["reasons"] = reasons
        if note:
            line["note"] = note
        found = zones.line_problems(line)
        if found:
            return found
        with self.lock:
            self.images[image_id] = {**self.images.get(image_id, {}), "line": line}
            zones.save(self.images, self.path)
        return []


def write_ticked(checker: LineChecker, out: Path) -> list[Path]:
    """The checked lines of round 2's sample drawn on their pictures."""
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for k in zones.round2_sample(checker.views, checker.images):
        line = checker.images.get(k, {}).get("line")
        if line is None:
            continue
        with Image.open(checker.pictures / f"{k}.png") as img:
            ticked(img, line["rim"], line["centre"]).save(out / f"{k}.png")
        written.append(out / f"{k}.png")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="check round 2's drawn lines (ADR 0006, s.9)")
    parser.add_argument("--draft", type=int, nargs="?", const=DRAFT, help="draft the first N")
    parser.add_argument("--ticked", action="store_true", help="draw the checked lines")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = parser.parse_args(argv)
    if args.draft is not None:
        images, views = zones.load(), labels.load(labels.LABELS_TOML)
        ids = zones.round2_pool(views, images)[: args.draft]
        tracemalloc.start()
        start = time.perf_counter()
        drafts = draft(ids)
        peak = tracemalloc.get_traced_memory()[1] / 2**20
        for k, d in drafts.items():
            support = f" support {d['support']}/{d['paths']}" if "paths" in d else ""
            print(f"{k[:40]:40s} {d['source']:13s}{support}")
        print(f"drafted {len(drafts)} in {time.perf_counter() - start:.0f} s, peak {peak:.0f} MiB")
        return 0
    checker = LineChecker()
    if args.ticked:
        written = write_ticked(checker, OUT_DIR / "ticked")
        print(f"drew {len(written)} checked lines into {OUT_DIR / 'ticked'}")
        return 0
    if not checker.sample:
        print(f"nothing to check: no drafts in {OUT_DIR}; run with --draft first")
        return 1
    server = ThreadingHTTPServer((HOST, args.port), handler(checker))
    url = f"http://{HOST}:{args.port}/"
    print(f"line check at {url} ({len(checker.sample)} drafted faces); Ctrl+C to stop")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
