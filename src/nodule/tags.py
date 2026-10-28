"""Re-tagging the reference library with the characterization schema (ADR 0004,
section 8).

The library and its photos live in agatesim (`references/library*`), which is
found next to this repository or at `NODULE_AGATESIM`. agatesim's own code
resolves those paths from where it is installed, so the paths are passed in
from here. No image is copied: `references/library-tags.toml` holds only
derived values, keyed by agatesim library id.

Each usable photo gets one record:

- agatesim's labels carried over as aliases (section 7): verdict, forms,
  features;
- the view (section 1): kind, finish and lighting are "unknown" until tagged
  (choice 9b), and the library has no scale;
- C1-C3 measured here on the outline that agatesim's segmentation finds
  (`nodule.outline`);
- C7, C8, C9, C11 and C17 read from agatesim's cached band-shape statistics
  (`agatesim library shape`), where the cache is current and the photo passes
  agatesim's shape gate ("fit" or "borderline"); "unknown" otherwise.

A value that cannot be had is written "unknown", never left out.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import tracemalloc
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import tomli_w
from agatesim import progress
from agatesim.library import Label, LibraryImage, load_labels, load_library
from agatesim.shapes import VERSION as SHAPES_VERSION
from agatesim.shapes import WORKING_PX
from agatesim.specimen.photo import load_photo
from agatesim.specimen.segment import segment

from nodule import outline

ROOT = Path(__file__).resolve().parents[2]
TAGS_TOML = ROOT / "references" / "library-tags.toml"
UNKNOWN = "unknown"
# The library's section-sampling model (ADR 0004, choice 9a).
SAMPLING = "unknown, compared as centre-seeking"
# Core descriptors read from agatesim's cached statistics: name here -> name there.
IMPORTED = {
    "C7_spread": "spread",
    "C8_trend": "trend",
    "C9_ratio": "ratio",
    "C11_hurst": "hurst",
    "C17_core": "core",
}
GATE_PASSES = ("fit", "borderline")

HEADER = """\
# Reference library tags under the characterization schema (ADR 0004).
# Written by `python -m nodule.tags`; do not edit by hand. Keyed by agatesim
# library id; no image is held here. "unknown" marks a value that cannot be
# had from this photo, or has not been tagged yet. Appearance only: never
# evidence for a formation mechanism (ADR 0002).

"""

Value = float | int | str | bool | list[str]


def agatesim_dir() -> Path:
    """agatesim's folder: `NODULE_AGATESIM`, or next to this repository."""
    return Path(os.environ.get("NODULE_AGATESIM", ROOT.parent / "agatesim"))


def image_path(agatesim: Path, image: LibraryImage) -> Path:
    return agatesim / "references" / "library" / image.path.name


def imported(entry: dict[str, object] | None, size: int) -> dict[str, Value]:
    """The five descriptors from one cached agatesim entry, where the cache is
    current for this file and the photo passes the shape gate."""
    current = (
        entry is not None
        and str(entry.get("key", "")).split(":")[:2] == [str(SHAPES_VERSION), str(size)]
        and entry.get("level") in GATE_PASSES
    )
    out: dict[str, Value] = {}
    for name, there in IMPORTED.items():
        value = entry.get(there) if entry is not None and current else None
        number = isinstance(value, int | float) and not isinstance(value, bool)
        out[name] = round(float(value), 4) if number else UNKNOWN  # pyright: ignore[reportArgumentType]
    return out


def record(
    label: Label, shape: outline.Outline | None, entry: dict[str, object] | None, size: int
) -> dict[str, Value]:
    """One photo's tags."""
    rec: dict[str, Value] = {
        "verdict": label.verdict,
        "forms": list(label.forms),
        "features": list(label.features),
        "view_kind": UNKNOWN,
        "finish": UNKNOWN,
        "lighting": UNKNOWN,
        "scale": "none",
        "shape_gate": str(entry.get("level", UNKNOWN)) if entry is not None else UNKNOWN,
    }
    if shape is None:
        rec |= {k: UNKNOWN for k in ("C1_aspect", "C1_regularity", "C2_circularity")}
        rec |= {"C3_irregularity": UNKNOWN, "C3_concavities": UNKNOWN}
    else:
        rec |= {
            "C1_aspect": round(shape.aspect, 4),
            "C1_regularity": round(shape.regularity, 4),
            "C2_circularity": round(shape.circularity, 4),
            "C3_irregularity": round(shape.irregularity, 4),
            "C3_concavities": shape.concavities,
        }
    return rec | imported(entry, size)


def measure_outline(data: bytes) -> outline.Outline | None:
    """C1-C3 of a photo's specimen outline, at agatesim's working size."""
    seg = segment(load_photo(data, WORKING_PX).rgb)
    if len(seg.outline) < 3:
        return None
    return outline.measure(seg.mask)


def write(records: dict[str, dict[str, Value]], path: Path = TAGS_TOML) -> None:
    doc = {
        "meta": {"schema": "ADR 0004", "sampling": SAMPLING, "photos": len(records)},
        "images": dict(sorted(records.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + tomli_w.dumps(doc), "utf-8")


def tag_library(
    agatesim: Path, sample: int | None = None
) -> tuple[dict[str, dict[str, Value]], list[float]]:
    """Tags of the usable photos (or a fixed random `sample` of them), and the
    seconds each took."""
    refs = agatesim / "references"
    library = load_library(refs / "library.toml")
    labels = load_labels(refs / "library-labels.toml")
    shapes_json = refs / "models" / "shapes.json"
    cache: dict[str, dict[str, object]] = (
        json.loads(shapes_json.read_text("utf-8")) if shapes_json.exists() else {}
    )
    ids = sorted(k for k, v in labels.items() if v.verdict != "unusable")
    if sample is not None:
        ids = sorted(np.random.default_rng(0).choice(ids, size=sample, replace=False).tolist())
    records: dict[str, dict[str, Value]] = {}
    seconds: list[float] = []
    for n, image_id in enumerate(ids):
        progress.report(n / max(len(ids), 1), "tagging the library")
        path = image_path(agatesim, library.get(image_id))
        if not path.exists():
            continue
        start = time.perf_counter()
        data = path.read_bytes()
        shape = measure_outline(data)
        records[image_id] = record(labels[image_id], shape, cache.get(image_id), len(data))
        seconds.append(time.perf_counter() - start)
    return records, seconds


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="re-tag the reference library (ADR 0004)")
    parser.add_argument("--sample", type=int, help="tag only this many photos, chosen at random")
    parser.add_argument("--out", type=Path, default=TAGS_TOML, help="where to write the tags")
    args = parser.parse_args(argv)
    agatesim = agatesim_dir()
    if not (agatesim / "references" / "library.toml").exists():
        print(f"no agatesim library at {agatesim}; set NODULE_AGATESIM")
        return 1
    tracemalloc.start()
    start = time.perf_counter()
    records, seconds = tag_library(agatesim, args.sample)
    total = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    write(records, args.out)
    per = np.array(seconds) if seconds else np.zeros(1)
    print(f"tagged {len(records)} photos into {args.out}")
    print(
        f"{total:.1f} s in all; per photo median {np.median(per):.2f} s, max {per.max():.2f} s;"
        f" peak traced memory {peak / 2**20:.0f} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
