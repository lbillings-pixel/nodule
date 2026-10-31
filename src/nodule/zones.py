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

**Round 2** (section 9, choices 7l-7o; rows carry `round = 2`, rows without
it are round 1 and keep its rules and scoring): C24 is faint striations only,
per third (`ROUND2_CUE_FIELDS`); C25's zones are read along the face's drawn
line (`nodule.zoneline`), each at least `MIN_SHARE` of it; a face's C25
agrees where the kinds match on at least `LINE_AGREE` of the line. The owner's
checked line is stored as `line` beside the labels (`line_problems`). The
blind 10 are drawn from banded faces nobody has checked (`round2_pool`, choice
7m): the next in the pool stands in for a face the owner marks not banded at
the line check.

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
ZONE_KINDS = (
    "even bands",
    "curdled or mottled bands",
    "plain: no visible bands",
    "coarse quartz",
    "cavity",
)
MAX_CENTRES = 50
SHARE_SUM = 0.05  # shares may sum to 1 within this
BLIND = 10  # of the check sample, answered with the proposals hidden (choice 7f)
RECOUNT = 5  # faces the owner recounts by eye after a delay (choice 7h)
RECOUNT_FROM = date(2026, 11, 5)  # a week after the first eye counts (2026-10-29)
SEED = 6  # ADR 0006; its spawned streams pick the blind and recount faces
FIELDS = ("banded", "C23_centres", *CUE_FIELDS, "C25_zones")

# Round 2 (section 9).
ROUND2_CUE_FIELDS = tuple(f"C24_{t}_striations" for t in THIRDS)  # choice 7n
ROUND2_FIELDS = ("round", "banded", "C23_centres", *ROUND2_CUE_FIELDS, "C25_zones")
MIN_SHARE = 0.05  # choice 7l: a zone covers at least this share of the line
LINE_AGREE = 2 / 3  # choice 7o: C25 agrees where kinds match on this much of the line
POOL_STREAM = 3  # child of SEED's streams that orders round 2's pool (choice 7m)
POOL_VIEWS = ("cut face", "slab in transmitted light")  # views that can show a banded face
LINE_REASONS = ("crack or edge", "misses the bands", "waterline banding")
LINE_SOURCES = ("paths", "deepest point", "owner")
MIN_OUTLINE = 3  # corners of an outline the owner traced where the face was wrong

Row = dict[str, Any]


def round_of(row: Mapping[str, object]) -> int:
    return 2 if row.get("round") == 2 else 1


def problems(row: Mapping[str, object], checked: bool = False) -> list[str]:
    """What is wrong with one photo's C23-C25 labels (empty when nothing is)."""
    second = "round" in row
    fields = ROUND2_FIELDS if second else FIELDS
    cue_fields = ROUND2_CUE_FIELDS if second else CUE_FIELDS
    allowed = set(fields) | {"note"} | ({"blind"} if checked else {"confidence"})
    out = [f"{k}: not a label" for k in row if k not in allowed]
    if second and (type(row["round"]) is not int or row["round"] != 2):
        out.append(f"round: {row['round']!r} is not 2 (round 1 rows carry no round)")
    if row.get("banded") not in ("yes", "no"):
        out.append(f"banded: {row.get('banded')!r} is not yes or no")
    centres = row.get("C23_centres")
    if centres != UNKNOWN and (
        isinstance(centres, bool) or not isinstance(centres, int) or not 0 <= centres <= MAX_CENTRES
    ):
        out.append(
            f"C23_centres: {centres!r} is not a whole number from 0 to {MAX_CENTRES}, or unknown"
        )
    for name in cue_fields:
        if row.get(name) not in CUE_VALUES:
            out.append(f"{name}: {row.get(name)!r} is not one of {', '.join(CUE_VALUES)}")
    out += zone_problems(row.get("C25_zones"), MIN_SHARE if second else 0.0)
    if not checked and row.get("confidence") not in labels.PROPOSED_ONLY["confidence"]:
        out.append(f"confidence: {row.get('confidence')!r} is not high, medium or low")
    if checked and not isinstance(row.get("blind", False), bool):
        out.append("blind: not true or false")
    if not isinstance(row.get("note", ""), str):
        out.append("note: not text")
    unknown = centres == UNKNOWN and not row.get("C25_zones")
    unknown = unknown and all(row[n] == UNKNOWN for n in cue_fields)
    if not out and row["banded"] == "no" and not unknown:
        out.append("banded: no, so C23-C25 are unknown")
    return out


def zone_problems(found: object, min_share: float = 0.0) -> list[str]:
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
        elif share < min_share - 1e-9:
            out.append(f"C25 zone {i}: share {share} is under {min_share} of the line (choice 7l)")
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
    return np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(child,)))


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


def round2_pool(
    view_labels: Mapping[str, Mapping[str, labels.Row]], images: Mapping[str, Mapping[str, Row]]
) -> list[str]:
    """Round 2's candidates in their fixed order (choice 7m): photos whose view
    can show a banded face (`POOL_VIEWS`), outside the round 1 check sample
    and with no round 1 labels, shuffled by stream `POOL_STREAM`."""
    sample = set(labels.check_sample(view_labels))
    pool = sorted(
        k
        for k, e in view_labels.items()
        if labels.chosen(e)[0].get("view_kind") in POOL_VIEWS
        and k not in sample
        and all(round_of(r) == 2 for n, r in images.get(k, {}).items() if n != "line")
    )
    return [str(i) for i in stream(POOL_STREAM).permutation(pool)]


def round2_sample(
    view_labels: Mapping[str, Mapping[str, labels.Row]],
    images: Mapping[str, Mapping[str, Row]],
    size: int = BLIND,
) -> list[str]:
    """The first `size` of the pool the owner has not marked not banded at the
    line check: the blind faces of round 2."""
    pool = round2_pool(view_labels, images)
    kept = [k for k in pool if images.get(k, {}).get("line", {}).get("banded") != "no"]
    return kept[:size]


def is_point(p: object) -> bool:
    """Whether `p` is [x, y], each a fraction of the picture from 0 to 1."""
    return (
        isinstance(p, list)
        and len(cast("list[object]", p)) == 2
        and all(
            isinstance(v, int | float) and not isinstance(v, bool) and 0 <= v <= 1
            for v in cast("list[object]", p)
        )
    )


def line_problems(line: Mapping[str, object]) -> list[str]:
    """What is wrong with the owner's checked line of one face: either
    `banded = "no"` (the face shows no banded face; the pool's next stands
    in), or rim and centre as [x, y] fractions of the picture, the centre's
    source, the direction draw and one reason (`LINE_REASONS`) per draw
    passed over (choice 7o). `outline`, where the computer's face was wrong,
    is the stone's edge as the owner traced it ([x, y] fractions in order);
    the rim end then comes from it."""
    allowed = (
        {"banded", "note"}
        if "banded" in line
        else {"rim", "centre", "source", "draw", "reasons", "outline", "note"}
    )
    out = [f"line {k}: not a field" for k in line if k not in allowed]
    if not isinstance(line.get("note", ""), str):
        out.append("line note: not text")
    if "banded" in line:
        if line["banded"] != "no":
            out.append(f"line banded: {line['banded']!r}; only no is stored")
        return out
    for end in ("rim", "centre"):
        p = line.get(end)
        if not is_point(p):
            out.append(f"line {end}: {p!r} is not [x, y], each from 0 to 1")
    if "outline" in line:
        o = line["outline"]
        if not (
            isinstance(o, list)
            and len(cast("list[object]", o)) >= MIN_OUTLINE
            and all(is_point(p) for p in cast("list[object]", o))
        ):
            out.append(f"line outline: at least {MIN_OUTLINE} points [x, y], each from 0 to 1")
    if line.get("source") not in LINE_SOURCES:
        out.append(f"line source: {line.get('source')!r} is not one of {', '.join(LINE_SOURCES)}")
    draw, reasons = line.get("draw"), line.get("reasons", [])
    if type(draw) is not int or draw < 0:
        out.append(f"line draw: {draw!r} is not a whole number from 0")
    elif not isinstance(reasons, list) or len(cast("list[object]", reasons)) != draw:
        out.append(f"line reasons: one per draw passed over ({draw}) is needed")
    elif any(r not in LINE_REASONS for r in cast("list[object]", reasons)):
        out.append(f"line reasons: each one of {', '.join(LINE_REASONS)}")
    return out


def items(proposed: Mapping[str, Any], checked: Mapping[str, Any]) -> list[tuple[str, bool]]:
    """The scored items of one face: (descriptor, agreed). C23 is one item,
    each C24 cue one, each C25 zone position one (a missing zone disagrees;
    two unknown sequences agree as one item). Only `banded` is scored when
    either side says the face is not banded. Round 2 (choices 7n, 7o): each
    striation third one item, C25 one item, agreed where `line_match` is at
    least `LINE_AGREE`."""
    second = round_of(checked) == 2
    if round_of(proposed) != round_of(checked):
        raise ValueError("a round 1 answer cannot be scored against a round 2 one")
    out = [("banded", proposed["banded"] == checked["banded"])]
    if proposed["banded"] != "yes" or checked["banded"] != "yes":
        return out
    out.append(("C23", proposed["C23_centres"] == checked["C23_centres"]))
    out += [
        ("C24", proposed[n] == checked[n]) for n in (ROUND2_CUE_FIELDS if second else CUE_FIELDS)
    ]
    p, c = proposed["C25_zones"], checked["C25_zones"]
    if not p and not c:
        return [*out, ("C25", True)]
    if second:
        return [*out, ("C25", line_match(p, c) >= LINE_AGREE - 1e-9)]
    for i in range(max(len(p), len(c))):
        same = i < len(p) and i < len(c) and p[i]["kind"] == c[i]["kind"]
        out.append(("C25", same))
    return out


def line_match(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]) -> float:
    """The share of the line, rim (0) to centre (1), where two zone sequences
    give the same kind; 0 when either is unknown (empty). Shares are scaled
    to sum to 1, since they may miss it by `SHARE_SUM`."""
    if not a or not b:
        return 0.0

    def stretches(found: Sequence[Mapping[str, Any]]) -> list[tuple[float, float, str]]:
        total = sum(z["share"] for z in found)
        ends = np.cumsum([z["share"] / total for z in found])
        starts = np.concatenate([[0.0], ends[:-1]])
        return [
            (float(s), float(e), z["kind"]) for s, e, z in zip(starts, ends, found, strict=True)
        ]

    return sum(
        max(0.0, min(e1, e2) - max(s1, s2))
        for s1, e1, k1 in stretches(a)
        for s2, e2, k2 in stretches(b)
        if k1 == k2
    )


def share_gaps(proposed: Mapping[str, Any], checked: Mapping[str, Any]) -> list[float]:
    """|share difference| of C25 zones of the same kind at the same position."""
    pairs = zip(proposed.get("C25_zones", []), checked.get("C25_zones", []), strict=False)
    return [abs(p["share"] - c["share"]) for p, c in pairs if p["kind"] == c["kind"]]


# Parts the fallback test is applied to, each alone and all pooled (choice 7j),
# so C24's 12 items per face can neither trip it alone unseen nor hide the rest.
TESTED = ("C23", "C24", "C25", "whole")


def report(images: Mapping[str, Mapping[str, Row]]) -> str:
    """Agreement of the owner's check with the proposals, blind and shown apart,
    and the fallback test of ADR 0006 section 6: more than a third disagree,
    in any of `TESTED`. Each round is scored and tested apart, round 2 only
    once it has answers."""
    done = [e for e in images.values() if {"proposed", "checked"} <= e.keys()]
    first = [e for e in done if round_of(e["checked"]) == 1]
    second = [e for e in done if round_of(e["checked"]) == 2]
    lines = agreement(first, "agreement with the proposals, item by item:")
    if second:
        lines += agreement(second, "round 2 (ADR 0006, s.9), item by item:")
    return "\n".join(lines)


def agreement(entries: Sequence[Mapping[str, Row]], title: str) -> list[str]:
    groups: dict[str, list[tuple[str, bool]]] = {"blind": [], "shown": []}
    gaps: list[float] = []
    matches: list[float] = []
    for e in entries:
        group = "blind" if e["checked"].get("blind") else "shown"
        groups[group] += items(e["proposed"], e["checked"])
        if round_of(e["checked"]) == 1:
            gaps += share_gaps(e["proposed"], e["checked"])
        elif "yes" == e["proposed"]["banded"] == e["checked"]["banded"]:
            matches.append(line_match(e["proposed"]["C25_zones"], e["checked"]["C25_zones"]))
    groups["all"] = groups["blind"] + groups["shown"]
    lines = [title]
    for name, scored in groups.items():
        parts = [
            f"{d} {sum(a for k, a in scored if k == d)}/{sum(k == d for k, _ in scored)}"
            for d in ("banded", "C23", "C24", "C25")
        ]
        same = sum(a for _, a in scored)
        lines.append(f"{name:6s} {same}/{len(scored)} items ({', '.join(parts)})")
    for name in ("blind", "all"):
        for part in TESTED:
            scored = [a for k, a in groups[name] if part == "whole" or k == part]
            if scored and sum(not a for a in scored) > len(scored) / 3:
                lines.append(
                    f"FALLBACK: more than a third of the {name} {part} items disagree"
                    " (ADR 0006, s.6, choice 7j)"
                )
    if gaps:
        lines.append(
            f"C25 share gap where kinds agree: median {np.median(gaps):.2f}, n {len(gaps)}"
        )
    if matches:
        lines.append(
            f"C25 share of the line where kinds match: median {np.median(matches):.2f},"
            f" n {len(matches)}"
        )
    return lines


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
# Rows with round = 2 follow section 9; "line" is the owner's checked
# rim-to-centre line of a round 2 face (choice 7o). Appearance only: never
# evidence for a formation mechanism (ADR 0002).

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
