"""Labels of the library photos that are judged by eye (ADR 0004, sections 1
and 8, choice 9b).

View kind, finish, lighting, C4 (exterior surface), C5 (rind), C22 (fracture
traces) and the outline check ("does the drawn outline follow the stone's
rim?") cannot be measured from the photo, so they are labelled. Per choice 9b,
Claude proposes them for every usable photo with helper agents, looking at the
outline-check pictures (Commons photos only), and the owner checks a fixed
random sample (`CHECK_SAMPLE`) on the check page (`nodule.check`). An owner's
check overrides the proposal; agreement between the two on the sample says how
far the remaining proposals can be trusted.

The owner may also give a rough band count by eye (`C6_eye`): light bands
crossed from the rim to the centre, as C6 counts them. It is a comparison with
the eye, not a true count.

Labels live in `references/library-view-labels.toml`, keyed by agatesim library
id, each photo with a `proposed` table and, once checked, a `checked` table.
Appearance only: never evidence for a formation mechanism (ADR 0002).
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import tomli_w

LABELS_TOML = Path(__file__).resolve().parents[2] / "references" / "library-view-labels.toml"
UNKNOWN = "unknown"
CHECK_SAMPLE = 30
PROPOSED_BY = "Claude (helper agents viewing the outline-check pictures)"

# Labels with a fixed set of values (ADR 0004 section 1 and the C4, C5, C22 rows).
CHOICES: dict[str, tuple[str, ...]] = {
    "view_kind": (
        "cut face",
        "exterior",
        "slab in transmitted light",
        "broken surface",
        UNKNOWN,
    ),
    "finish": ("sawn", "polished", "natural", "tumbled", UNKNOWN),
    "lighting": ("reflected", "transmitted", UNKNOWN),
    "C4_surface": (
        "smooth",
        "blistered",
        "ridged",
        "lobed",
        "coarse-grained",
        "broken",
        UNKNOWN,
    ),
    # Adhering host rock is labelled host, not rind (C5's negative control).
    "C5_rind": (
        "absent",
        "clay or celadonite coat",
        "spherulitic shell",
        "adhering host",
        UNKNOWN,
    ),
    "C22_fill": ("none", "open", "filled", "both", UNKNOWN),
    "C22_offset": ("yes", "no", UNKNOWN),
    "outline": ("yes", "no"),
}
# Labels that are a number or "unknown": name -> (lowest, highest, whole number).
NUMBERS: dict[str, tuple[float, float, bool]] = {
    "C5_width": (0.0, 0.5, False),  # share of the face's width
    "C22_count": (0, 1000, True),
}
FIELDS = (*CHOICES, *NUMBERS)
# Fields one side only gives.
PROPOSED_ONLY = {"confidence": ("high", "medium", "low")}
EYE_COUNT = (0, 1000)

Value = str | int | float
Row = dict[str, Value]


def problems(row: Mapping[str, object], checked: bool = False) -> list[str]:
    """What is wrong with one photo's labels (an empty list when nothing is)."""
    out: list[str] = []
    allowed = set(FIELDS) | {"note"}
    allowed |= {"C6_eye"} if checked else set(PROPOSED_ONLY)
    out += [f"{k}: not a label" for k in row if k not in allowed]
    for name, values in (CHOICES | ({} if checked else PROPOSED_ONLY)).items():
        if row.get(name) not in values:
            out.append(f"{name}: {row.get(name)!r} is not one of {', '.join(values)}")
    numbers = NUMBERS | ({"C6_eye": (*EYE_COUNT, True)} if checked else {})
    for name, (low, high, whole) in numbers.items():
        value = row.get(name, UNKNOWN if name == "C6_eye" else None)
        if value == UNKNOWN:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not low <= value <= high
            or (whole and not float(value).is_integer())
        ):
            out.append(
                f"{name}: {value!r} is not {'a whole number' if whole else 'a number'}"
                f" from {low} to {high}, or unknown"
            )
    if not isinstance(row.get("note", ""), str):
        out.append("note: not text")
    if out:
        return out
    if row["C4_surface"] != UNKNOWN and row["view_kind"] != "exterior":
        out.append("C4_surface: an exterior surface class on a view that is not an exterior")
    if row["C5_width"] != UNKNOWN and row["C5_rind"] in ("absent", UNKNOWN):
        out.append("C5_width: a rind width where there is no rind")
    if row["C22_count"] == 0 and (row["C22_fill"], row["C22_offset"]) != ("none", UNKNOWN):
        out.append("C22: no fractures, so fill is none and offset unknown")
    if row["C22_fill"] == "none" and row["C22_count"] != 0:
        out.append("C22_fill: none, but the count is not 0")
    return out


def check_sample(ids: Iterable[str], size: int = CHECK_SAMPLE) -> list[str]:
    """The fixed random sample the owner checks."""
    pool = sorted(ids)
    picked = np.random.default_rng(0).choice(pool, size=min(size, len(pool)), replace=False)
    return sorted(str(i) for i in picked)


def load(path: Path = LABELS_TOML) -> dict[str, dict[str, Row]]:
    """The labels, by id: {"proposed": ..., "checked": ...} (either may be missing)."""
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text("utf-8")).get("images", {})


HEADER = """\
# Labels of the library photos judged by eye (ADR 0004, choice 9b). Keyed by
# agatesim library id; no image is held here. "proposed" is Claude's, from
# helper agents viewing the outline-check pictures; "checked" is the owner's,
# from `python -m nodule.check`, and overrides it. Appearance only: never
# evidence for a formation mechanism (ADR 0002).

"""


def save(images: Mapping[str, Mapping[str, Row]], path: Path = LABELS_TOML) -> None:
    sample = check_sample(images)
    doc = {
        "meta": {
            "schema": "ADR 0004, choice 9b",
            "proposed_by": PROPOSED_BY,
            "checked_by": "owner",
            "check_sample": sample,
        },
        "images": {k: dict(images[k]) for k in sorted(images)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + tomli_w.dumps(doc), "utf-8")


def chosen(entry: Mapping[str, Row] | None) -> tuple[Row, str]:
    """The labels that count for a photo, and who gave them: the owner's check
    where there is one, else Claude's proposal, else nothing."""
    if entry is None:
        return {}, UNKNOWN
    if "checked" in entry:
        return entry["checked"], "owner, checked"
    if "proposed" in entry:
        return entry["proposed"], "Claude, proposed"
    return {}, UNKNOWN


def agreement(images: Mapping[str, Mapping[str, Row]]) -> dict[str, tuple[int, int]]:
    """Per label, how many checked photos the owner left as proposed, of how many."""
    out: dict[str, tuple[int, int]] = {}
    pairs = [
        (e["proposed"], e["checked"])
        for e in images.values()
        if {"proposed", "checked"} <= e.keys()
    ]
    for name in FIELDS:
        same = sum(p.get(name) == c.get(name) for p, c in pairs)
        out[name] = (same, len(pairs))
    return out
