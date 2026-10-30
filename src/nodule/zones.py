"""Band centres, fine-banding cues and zones, judged by eye (ADR 0006, section 3).

- **C23** `C23_centres`: separate band centres in the face (0 when bands show
  but no centre lies in the face), or "unknown".
- **C24** `C24_<third>_<cue>`: for the outer, middle and inner third of the
  rim-to-centre distance, each cue present, absent or unknown: (a) colour
  gradient within a band, (b) faint striations at the limit, (c) shadow-like
  bands in radial sectors, (d) iris colours. A cue is a sign, never a count.
- **C25** `C25_zones`: the zones from rim to centre, each a kind and its share
  of the radius (shares sum to 1); an empty list means unknown.

`banded` says whether the photo shows a banded face at all; when it is "no",
C23-C25 are left unknown. Helper agents propose these labels from the
outline-check pictures and, on the check sample, from the owner's notes; the
owner checks them on `nodule.zonecheck` (choice 7b), the first `BLIND` of the
sample with the proposals hidden (choice 7f). Agreement is scored item by item
(`items`). Labels live in `references/band-zone-labels.toml`, keyed by agatesim
library id, as `proposed` and `checked` tables, like `nodule.labels`.

The control faces (`CONTROLS`) are synthetic, drawn here, and labelled by the
helpers among the real pictures without being told which they are: the
negative controls of C23-C25 (ADR 0006, section 3). Appearance only: never
evidence for a formation mechanism (ADR 0002).
"""

from __future__ import annotations

import argparse
import itertools
import json
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import tomli_w
from PIL import Image

from nodule import labels
from nodule._scipy import nearest

ZONES_TOML = labels.LABELS_TOML.with_name("band-zone-labels.toml")
CONTROL_DIR = Path(__file__).resolve().parents[2] / "runs" / "zone-controls"
UNKNOWN = labels.UNKNOWN
PROPOSED_BY = "Claude (helper agents viewing the outline-check pictures and the owner's notes)"

THIRDS = ("outer", "middle", "inner")
CUES = {
    "gradient": "(a) colour gradient within a band",
    "striations": "(b) faint striations at the limit",
    "shadow": "(c) shadow-like bands in radial sectors",
    "iris": "(d) iris colours",
}
CUE_FIELDS = tuple(f"C24_{t}_{c}" for t in THIRDS for c in CUES)
CUE_VALUES = ("present", "absent", UNKNOWN)
ZONE_KINDS = ("even bands", "curdled or mottled bands", "colour change", "coarse quartz", "cavity")
MAX_CENTRES = 50
SHARE_SUM = 0.05  # shares may sum to 1 within this
BLIND = 10  # of the check sample, answered with the proposals hidden (choice 7f)
RECOUNT = 5  # faces the owner recounts by eye after a delay (choice 7h)
RECOUNT_FROM = date(2026, 11, 5)  # a week after the first eye counts (2026-10-29)
SEED = 6  # ADR 0006; its spawned streams pick the blind and recount faces
FIELDS = ("banded", "C23_centres", *CUE_FIELDS, "C25_zones")

Row = dict[str, Any]


def problems(row: Mapping[str, object], checked: bool = False) -> list[str]:
    """What is wrong with one photo's C23-C25 labels (empty when nothing is)."""
    allowed = set(FIELDS) | {"note"} | ({"blind"} if checked else {"confidence"})
    out = [f"{k}: not a label" for k in row if k not in allowed]
    if row.get("banded") not in ("yes", "no"):
        out.append(f"banded: {row.get('banded')!r} is not yes or no")
    centres = row.get("C23_centres")
    if centres != UNKNOWN and (
        isinstance(centres, bool) or not isinstance(centres, int) or not 0 <= centres <= MAX_CENTRES
    ):
        out.append(
            f"C23_centres: {centres!r} is not a whole number from 0 to {MAX_CENTRES}, or unknown"
        )
    for name in CUE_FIELDS:
        if row.get(name) not in CUE_VALUES:
            out.append(f"{name}: {row.get(name)!r} is not one of {', '.join(CUE_VALUES)}")
    out += zone_problems(row.get("C25_zones"))
    if not checked and row.get("confidence") not in labels.PROPOSED_ONLY["confidence"]:
        out.append(f"confidence: {row.get('confidence')!r} is not high, medium or low")
    if checked and not isinstance(row.get("blind", False), bool):
        out.append("blind: not true or false")
    if not isinstance(row.get("note", ""), str):
        out.append("note: not text")
    unknown = centres == UNKNOWN and not row.get("C25_zones")
    unknown = unknown and all(row[n] == UNKNOWN for n in CUE_FIELDS)
    if not out and row["banded"] == "no" and not unknown:
        out.append("banded: no, so C23-C25 are unknown")
    return out


def zone_problems(found: object) -> list[str]:
    if not isinstance(found, list):
        return ["C25_zones: not a list of zones"]
    out: list[str] = []
    zones: list[dict[str, Any]] = []
    for i, zone in enumerate(cast("list[object]", found), 1):
        if not isinstance(zone, dict) or set(cast("dict[str, Any]", zone)) != {"kind", "share"}:
            out.append(f"C25 zone {i}: needs a kind and a share, nothing else")
            continue
        zone = cast("dict[str, Any]", zone)
        zones.append(zone)
        if zone["kind"] not in ZONE_KINDS:
            out.append(f"C25 zone {i}: {zone['kind']!r} is not one of {', '.join(ZONE_KINDS)}")
        share = zone["share"]
        if isinstance(share, bool) or not isinstance(share, int | float) or not 0 < share <= 1:
            out.append(f"C25 zone {i}: share {share!r} is not a number above 0, at most 1")
    if out or not zones:
        return out
    if any(a["kind"] == b["kind"] for a, b in itertools.pairwise(zones)):
        out.append("C25: two zones in a row of the same kind are one zone")
    total = sum(z["share"] for z in zones)
    if abs(total - 1) > SHARE_SUM:
        out.append(f"C25: shares sum to {total:.2f}, not 1")
    return out


def stream(child: int) -> np.random.Generator:
    """One of the spawned RNG streams that pick faces for the owner."""
    return np.random.default_rng(np.random.SeedSequence(SEED).spawn(2)[child])


def blind_sample(sample: Sequence[str], size: int = BLIND) -> list[str]:
    """The faces of the check sample answered with the proposals hidden."""
    pool = sorted(sample)
    return sorted(str(i) for i in stream(0).choice(pool, size=min(size, len(pool)), replace=False))


def recount_sample(
    view_labels: Mapping[str, Mapping[str, labels.Row]], size: int = RECOUNT
) -> list[str]:
    """Faces the owner gave an eye count on, to count again after a delay."""
    pool = sorted(
        k for k, e in view_labels.items() if isinstance(e.get("checked", {}).get("C6_eye"), int)
    )
    return sorted(str(i) for i in stream(1).choice(pool, size=min(size, len(pool)), replace=False))


def items(proposed: Mapping[str, Any], checked: Mapping[str, Any]) -> list[tuple[str, bool]]:
    """The scored items of one face: (descriptor, agreed). C23 is one item,
    each C24 cue one, each C25 zone position one (a missing zone disagrees;
    two unknown sequences agree as one item). Only `banded` is scored when
    either side says the face is not banded."""
    out = [("banded", proposed["banded"] == checked["banded"])]
    if proposed["banded"] != "yes" or checked["banded"] != "yes":
        return out
    out.append(("C23", proposed["C23_centres"] == checked["C23_centres"]))
    out += [("C24", proposed[n] == checked[n]) for n in CUE_FIELDS]
    p, c = proposed["C25_zones"], checked["C25_zones"]
    if not p and not c:
        return [*out, ("C25", True)]
    for i in range(max(len(p), len(c))):
        same = i < len(p) and i < len(c) and p[i]["kind"] == c[i]["kind"]
        out.append(("C25", same))
    return out


def share_gaps(proposed: Mapping[str, Any], checked: Mapping[str, Any]) -> list[float]:
    """|share difference| of C25 zones of the same kind at the same position."""
    pairs = zip(proposed.get("C25_zones", []), checked.get("C25_zones", []), strict=False)
    return [abs(p["share"] - c["share"]) for p, c in pairs if p["kind"] == c["kind"]]


def report(images: Mapping[str, Mapping[str, Row]]) -> str:
    """Agreement of the owner's check with the proposals, blind and shown apart,
    and the fallback test of ADR 0006 section 6 (more than a third disagree)."""
    groups: dict[str, list[tuple[str, bool]]] = {"blind": [], "shown": []}
    gaps: list[float] = []
    for e in images.values():
        if {"proposed", "checked"} <= e.keys():
            group = "blind" if e["checked"].get("blind") else "shown"
            groups[group] += items(e["proposed"], e["checked"])
            gaps += share_gaps(e["proposed"], e["checked"])
    groups["all"] = groups["blind"] + groups["shown"]
    lines = ["agreement with the proposals, item by item:"]
    for name, scored in groups.items():
        parts = [
            f"{d} {sum(a for k, a in scored if k == d)}/{sum(k == d for k, _ in scored)}"
            for d in ("banded", "C23", "C24", "C25")
        ]
        same = sum(a for _, a in scored)
        lines.append(f"{name:6s} {same}/{len(scored)} items ({', '.join(parts)})")
    for name in ("blind", "all"):
        scored = groups[name]
        if scored and sum(not a for _, a in scored) > len(scored) / 3:
            lines.append(
                f"FALLBACK: more than a third of the {name} items disagree (ADR 0006, s.6)"
            )
    if gaps:
        lines.append(
            f"C25 share gap where kinds agree: median {np.median(gaps):.2f}, n {len(gaps)}"
        )
    return "\n".join(lines)


def load(path: Path = ZONES_TOML) -> dict[str, dict[str, Row]]:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text("utf-8")).get("images", {})


HEADER = """\
# C23-C25 of the library photos, judged by eye (ADR 0006, section 3). Keyed by
# agatesim library id; no image is held here. "proposed" is Claude's, from
# helper agents viewing the outline-check pictures (and the owner's notes on
# the check sample); "checked" is the owner's, from `python -m nodule.zonecheck`,
# and overrides it; "recount" is the owner's later eye count (choice 7h).
# Appearance only: never evidence for a formation mechanism (ADR 0002).

"""


def save(images: Mapping[str, Mapping[str, Row]], path: Path = ZONES_TOML) -> None:
    doc = {
        "meta": {
            "schema": "ADR 0006, section 3",
            "proposed_by": PROPOSED_BY,
            "checked_by": "owner",
        },
        "images": {k: dict(images[k]) for k in sorted(images)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + tomli_w.dumps(doc), "utf-8")


def add_proposals(found: Mapping[str, Row], path: Path = ZONES_TOML) -> list[str]:
    """Store helper proposals; nothing is written if any is wrong."""
    out = [f"{k}: {p}" for k, row in found.items() for p in problems(row)]
    if out:
        return out
    images = load(path)
    for k, row in found.items():
        images[k] = {**images.get(k, {}), "proposed": dict(row)}
    save(images, path)
    return []


# Synthetic control faces. Drawn in world coordinates ([-1, 1] across the
# square picture), so any size gives the same face; bands are far coarser than
# a pixel, so no cue below the limit can show. name -> expected labels.
CONTROLS: dict[str, Row] = {
    "one-set": {"C23_centres": 1, "C25_zones": [{"kind": "even bands", "share": 1.0}]},
    "two-sets": {"C23_centres": 2, "C25_zones": [{"kind": "even bands", "share": 1.0}]},
    "zones": {
        "C23_centres": 1,
        "C25_zones": [
            {"kind": "even bands", "share": 0.5},
            {"kind": "coarse quartz", "share": 0.3},
            {"kind": "cavity", "share": 0.2},
        ],
    },
}
COUPLETS = 12  # light-dark band pairs from rim to centre in each control set
BACKGROUND = np.array([46, 44, 42], float)
MILKY = np.array([226, 226, 230], float)
CLEAR = np.array([104, 120, 146], float)


def render_control(name: str, size: int = 1024) -> np.ndarray:
    """One control face as an RGB picture, `size` pixels square."""
    if name not in CONTROLS:
        raise ValueError(f"no control named {name!r}")
    x, y = np.meshgrid(*[(np.arange(size) + 0.5) / size * 2 - 1] * 2)
    if name == "two-sets":
        # Two sets that meet in one outline with a waist.
        d = np.minimum(np.hypot((x + 0.42) / 0.5, y / 0.6), np.hypot((x - 0.42) / 0.5, y / 0.6))
    else:
        d = np.hypot(x / 0.85, y / 0.7)  # 1 at the rim, 0 at the centre
    band = 0.5 + 0.5 * np.cos(2 * np.pi * COUPLETS * (1 - d))
    rgb = CLEAR + (MILKY - CLEAR) * band[..., None]
    if name == "zones":
        quartz = (d < 0.5) & (d >= 0.2)
        # Facets: fixed points in world coordinates, nearest one shades the grain.
        points = np.random.default_rng(np.random.SeedSequence(SEED).spawn(3)[2]).uniform(
            -1, 1, (400, 2)
        )
        shade = np.linspace(0.85, 1.0, len(points))
        near = nearest(points, np.column_stack([x[quartz], y[quartz]]))
        rgb[quartz] = np.array([238, 236, 228]) * shade[near, None]
        rgb[d < 0.2] = [20, 18, 16]
    rgb[d > 1] = BACKGROUND
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def write_controls(out_dir: Path = CONTROL_DIR, size: int = 1024) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in CONTROLS:
        path = out_dir / f"{name}.png"
        Image.fromarray(render_control(name, size)).save(path)
        paths.append(path)
    return paths


def control_problems(name: str, row: Mapping[str, Any]) -> list[str]:
    """Where a labeller's answer on a control face misses what it must give."""
    want = CONTROLS[name]
    out: list[str] = []
    if row.get("banded") != "yes":
        out.append(f"{name}: banded {row.get('banded')!r}, not yes")
        return out
    if row["C23_centres"] != want["C23_centres"]:
        out.append(f"{name}: C23 {row['C23_centres']!r}, not {want['C23_centres']}")
    got = [z["kind"] for z in row["C25_zones"]]
    if got != [z["kind"] for z in want["C25_zones"]]:
        out.append(f"{name}: C25 kinds {got}")
    elif any(
        abs(a["share"] - b["share"]) > 0.1
        for a, b in zip(row["C25_zones"], want["C25_zones"], strict=True)
    ):
        out.append(f"{name}: C25 shares off by more than 0.1")
    cues = [n for n in CUE_FIELDS if row[n] == "present"]
    if cues:
        out.append(f"{name}: a cue on a face without fine bands: {', '.join(cues)}")
    return out


def read_helpers(paths: Iterable[Path]) -> dict[str, Row]:
    found: dict[str, Row] = {}
    for path in paths:
        found |= json.loads(Path(path).read_text("utf-8"))
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C23-C25 labels (ADR 0006, section 3)")
    parser.add_argument("--controls", action="store_true", help="draw the control faces")
    parser.add_argument(
        "--add", nargs="+", type=Path, metavar="JSON", help="store helper proposals"
    )
    parser.add_argument(
        "--control-map", type=Path, help="JSON: picture name in helper files -> control"
    )
    parser.add_argument(
        "--report", action="store_true", help="print agreement with the owner's check"
    )
    args = parser.parse_args(argv)
    if args.controls:
        for path in write_controls():
            print(path)
    if args.add:
        found = read_helpers(args.add)
        names: dict[str, str] = (
            json.loads(args.control_map.read_text("utf-8")) if args.control_map else {}
        )
        for alias, name in names.items():
            row = found.pop(alias)
            missed = problems(row) or control_problems(name, row)
            print(f"control {name}: " + ("; ".join(missed) if missed else "as it must be"))
        refused = add_proposals(found)
        print(
            "\n".join(refused) if refused else f"stored {len(found)} proposals in {ZONES_TOML.name}"
        )
        if refused:
            return 1
    if args.report:
        print(report(load()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
