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
- the view (section 1): kind, finish and lighting, labelled by eye with C4,
  C5, C22 and the outline check (`nodule.labels`, choice 9b); the library has
  no scale;
- C1-C3 measured here on the outline that agatesim's rim-finder draws
  (`nodule.outline`, `nodule.rims`), only where a helper or the owner has
  judged that very outline to follow the whole rim (agatesim ADR 0075);
- C7, C8, C9, C11 and C17 read from agatesim's cached band-shape statistics
  (`agatesim library shape`), where the cache is current and the photo passes
  agatesim's shape gate ("fit" or "borderline"); "unknown" otherwise;
- C6 and C10 measured here along rim-to-centre profiles (`nodule.bands`) on
  the same photos, with the band spacing and noise agatesim's cache holds:
  inside the judged rim outline, or else inside agatesim's working region
  (`segment()`), where C10 still counts.

Where the rim outline is missing, unjudged or judged "no", C1-C3 and C6 are
"unknown" (section 8). The labels' own outline check judged the old
`segment()` outline, so it no longer applies. `--relabel` applies new labels
to the tag file without measuring again.

A value that cannot be had is written "unknown", never left out.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import tomllib
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
from agatesim.specimen.quality import luminance
from agatesim.specimen.segment import segment

from nodule import bands, labels, outline, rims

ROOT = Path(__file__).resolve().parents[2]
TAGS_TOML = ROOT / "references" / "library-tags.toml"
UNKNOWN = labels.UNKNOWN
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
# Values that count only once the outline is seen to follow the rim (section 8).
OUTLINE_BOUND = (
    "C1_aspect",
    "C1_regularity",
    "C2_circularity",
    "C3_irregularity",
    "C3_concavities",
    "C6_count",
    "C6_low",
    "C6_high",
    "C6_rule_gap",
)
# Band count and persistence measured here: name in the record -> field of `bands.Bands`.
MEASURED = {
    "C6_count": "count",
    "C6_low": "count_low",
    "C6_high": "count_high",
    "C6_rule_gap": "rule_gap",
    "C6_profiles": "profiles",
    "C10_persistence": "persistence",
}

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


def passes(entry: dict[str, object] | None, size: int) -> bool:
    """The cached agatesim entry is current for this file and the photo passes
    the shape gate."""
    return (
        entry is not None
        and str(entry.get("key", "")).split(":")[:2] == [str(SHAPES_VERSION), str(size)]
        and entry.get("level") in GATE_PASSES
    )


def imported(entry: dict[str, object] | None, size: int) -> dict[str, Value]:
    """The five descriptors from one cached agatesim entry, where it `passes`."""
    current = passes(entry, size)
    out: dict[str, Value] = {}
    for name, there in IMPORTED.items():
        value = entry.get(there) if entry is not None and current else None
        number = isinstance(value, int | float) and not isinstance(value, bool)
        out[name] = round(float(value), 4) if number else UNKNOWN  # pyright: ignore[reportArgumentType]
    return out


def measured(band: bands.Bands | None) -> dict[str, Value]:
    """C6 and C10, "unknown" where they could not be judged."""
    out: dict[str, Value] = {}
    for name, field in MEASURED.items():
        value = getattr(band, field) if band is not None else None
        out[name] = round(float(value), 4) if value is not None else UNKNOWN
    if band is not None:
        out["C6_profiles"] = band.profiles
    return out


def record(
    label: Label,
    shape: outline.Outline | None,
    entry: dict[str, object] | None,
    size: int,
    band: bands.Bands | None = None,
    rim: dict[str, Value] | None = None,
) -> dict[str, Value]:
    """One photo's tags; `rim` from `rim_tags`. The outline-bound values count
    only where the rim outline was judged "yes"."""
    rec: dict[str, Value] = {
        "verdict": label.verdict,
        "forms": list(label.forms),
        "features": list(label.features),
        **{name: UNKNOWN for name in labels.FIELDS if name != "outline"},
        **(rim if rim is not None else rim_tags(UNKNOWN, None)),
        "labels_by": UNKNOWN,
        "C6_eye": UNKNOWN,
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
    rec |= imported(entry, size) | measured(band)
    if rec["outline_check"] != "yes":
        rec |= {name: UNKNOWN for name in OUTLINE_BOUND}
    return rec


def rim_tags(reason: str | None, judged: rims.Verdict | None) -> dict[str, Value]:
    """What the rim-finder gave ("found", or "none: " and its reason) and the
    judgment of that very outline, if any. `reason` "unknown": not run."""
    rim = UNKNOWN if reason == UNKNOWN else "found" if reason is None else f"none: {reason}"
    if reason is not None or judged is None:
        return {"rim": rim, "outline_check": UNKNOWN, "outline_by": UNKNOWN}
    return {"rim": rim, "outline_check": "yes" if judged.passed else "no", "outline_by": judged.by}


def labelled(rec: dict[str, Value], entry: dict[str, labels.Row] | None) -> dict[str, Value]:
    """A record with its photo's labels filled in. The labels' own outline check
    judged an older outline and is left out."""
    row, by = labels.chosen(entry)
    out = dict(rec)
    for name in labels.FIELDS:
        if name != "outline":
            out[name] = row.get(name, UNKNOWN)
    out["labels_by"] = by
    out["C6_eye"] = row.get("C6_eye", UNKNOWN)
    return out


def measure_photo(
    finder: rims.Finder,
    verdicts: rims.Verdicts,
    image_id: str,
    data: bytes,
    entry: dict[str, object] | None,
) -> tuple[outline.Outline | None, bands.Bands | None, dict[str, Value]]:
    """C1-C3 of the photo's rim outline where it was judged "yes" (at
    `rims.RIM_PX`), and C6 and C10 at full resolution where agatesim's cache
    `passes` and holds the band spacing and noise: inside that outline, or else
    inside agatesim's working region. Also the photo's `rim_tags`."""
    _, rim = finder.find(image_id, data)
    judged = rims.verdict(verdicts, image_id, rim.mask) if rim.found else None
    tag = rim_tags(rim.reason, judged)
    if judged is not None and judged.passed:
        mask, shape = rim.mask, outline.measure(rim.mask)
    else:
        mask, shape = segment(load_photo(data, WORKING_PX).rgb).mask, None
    if not mask.any():
        return shape, None, tag
    spacing = entry.get("spacing_px") if entry is not None else None
    noise = entry.get("noise") if entry is not None else None
    if not passes(entry, len(data)) or not isinstance(spacing, int | float):
        return shape, None, tag
    if not isinstance(noise, int | float):
        return shape, None, tag
    lum = luminance(load_photo(data, 10**6).rgb)
    scale = mask.shape[1] / lum.shape[1]
    band = bands.measure(
        lum, mask, scale, float(noise), float(spacing), rng=np.random.default_rng(0)
    )
    return shape, band, tag


def write(records: dict[str, dict[str, Value]], path: Path = TAGS_TOML) -> None:
    used = sum(rec.get("outline_check") == "yes" for rec in records.values())
    meta: dict[str, Value] = {"schema": "ADR 0004", "sampling": SAMPLING, "photos": len(records)}
    meta |= {"rim_outlines_used": used, "rim_outlines_wrong": rims.error_note()}
    doc = {
        "meta": meta,
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
    carried = load_labels(refs / "library-labels.toml")
    view = labels.load()
    shapes_json = refs / "models" / "shapes.json"
    cache: dict[str, dict[str, object]] = (
        json.loads(shapes_json.read_text("utf-8")) if shapes_json.exists() else {}
    )
    ids = sorted(k for k, v in carried.items() if v.verdict != "unusable")
    if sample is not None:
        ids = sorted(np.random.default_rng(0).choice(ids, size=sample, replace=False).tolist())
    finder = rims.Finder(agatesim)
    verdicts = rims.load()
    records: dict[str, dict[str, Value]] = {}
    seconds: list[float] = []
    for n, image_id in enumerate(ids):
        progress.report(n / max(len(ids), 1), "tagging the library")
        path = image_path(agatesim, library.get(image_id))
        if not path.exists():
            continue
        start = time.perf_counter()
        data = path.read_bytes()
        entry = cache.get(image_id)
        shape, band, rim = measure_photo(finder, verdicts, image_id, data, entry)
        rec = record(carried[image_id], shape, entry, len(data), band, rim)
        records[image_id] = labelled(rec, view.get(image_id))
        seconds.append(time.perf_counter() - start)
    return records, seconds


def relabel(path: Path = TAGS_TOML) -> int:
    """Apply the current labels to an existing tag file."""
    records: dict[str, dict[str, Value]] = tomllib.loads(path.read_text("utf-8"))["images"]
    view = labels.load()
    write({k: labelled(rec, view.get(k)) for k, rec in records.items()}, path)
    print(f"relabelled {len(records)} photos in {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="re-tag the reference library (ADR 0004)")
    parser.add_argument("--sample", type=int, help="tag only this many photos, chosen at random")
    parser.add_argument("--out", type=Path, default=TAGS_TOML, help="where to write the tags")
    parser.add_argument(
        "--relabel", action="store_true", help="apply the labels to the tag file, measuring nothing"
    )
    args = parser.parse_args(argv)
    if args.relabel:
        return relabel(args.out)
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
