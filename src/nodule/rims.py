"""Rim outlines of library photos, measured only after a yes (agatesim ADR 0075).

agatesim's rim-finder (agatesim ADRs 0073, 0074) draws the stone's outline
with its segmentation model, but about one outline in seven misses the rim.
So an outline counts only after a helper agent has judged that very outline
(the whole rim in the frame, and the outline following it); where the owner
has answered, the owner's answer wins.

A judgment belongs to the outline it was made on (agatesim ADR 0075, point 2).
Each verdict therefore carries a fingerprint of the judged outline: its mask
reduced to `MASK_PX` on the long side, as agatesim's `scripts/rim_check.py`
saved it, hashed. Here the outline is drawn again with the model passed in
explicitly, and a verdict applies only where the fingerprints match; a changed
outline (new agatesim commit or settings) finds no verdict and is not measured.

The verdicts are read from agatesim's `runs/rim-check/` (gitignored there)
and from this repository's own `runs/rim-check/` (gitignored), in the same
layout, into `references/rim-verdicts.toml` (`python -m nodule.rims`). Only
runs of the model (method C) are read: B's outlines would need their own
judgments. Outlines drawn here that nobody has judged yet are saved as a run
of their own, with pictures to judge (`--draw-unjudged NAME`); the helpers'
answers go to `judge-NAME.json` beside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tomli_w
from numpy.typing import NDArray
from PIL import Image

if TYPE_CHECKING:
    from agatesim.maps import PatchCache
    from agatesim.specimen.rim import Rim
    from agatesim.specimen.sam import Sam

ROOT = Path(__file__).resolve().parents[2]
VERDICTS_TOML = ROOT / "references" / "rim-verdicts.toml"
OWN_RUNS = ROOT / "runs" / "rim-check"
RIM_PX = 1024  # long side of the photo the outlines were drawn and judged on
MASK_PX = 256  # long side of the saved masks the fingerprint is taken from
# The owner's check of helper "yes" outlines (agatesim ADR 0075): wrong, checked.
OWNER_FOUND_WRONG = (2, 20)

HEADER = """\
# Judgments of agatesim's rim outlines on library photos (agatesim ADR 0075).
# Written by `python -m nodule.rims` from agatesim's runs/rim-check; do not
# edit by hand. Keyed by agatesim library id; `mask` fingerprints the judged
# outline, and a verdict applies only to an outline with the same fingerprint.
# No image is held here.

"""


@dataclass(frozen=True)
class Verdict:
    rim_in_frame: bool
    follows: bool
    by: str  # "helper" or "owner"
    run: str  # the agatesim rim-check run the outline was judged in

    @property
    def passed(self) -> bool:
        return self.rim_in_frame and self.follows


Verdicts = dict[str, dict[str, Verdict]]  # by photo, then by outline fingerprint


def fingerprint(mask: NDArray[np.bool_]) -> str:
    """The outline's mask reduced as rim_check.py saves it, then hashed."""
    small = Image.fromarray(np.asarray(mask, dtype=bool))
    small.thumbnail((MASK_PX, MASK_PX), Image.Resampling.NEAREST)
    return reduced_fingerprint(np.asarray(small, dtype=bool))


def reduced_fingerprint(small: NDArray[np.bool_]) -> str:
    """Fingerprint of a mask already reduced to `MASK_PX`."""
    h = hashlib.sha256(f"{small.shape[0]}x{small.shape[1]}:".encode())
    h.update(np.packbits(small).tobytes())
    return h.hexdigest()[:16]


def wilson(wrong: int, checked: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval of a share."""
    p = wrong / checked
    centre = (p + z * z / (2 * checked)) / (1 + z * z / checked)
    half = z * math.sqrt(p * (1 - p) / checked + z * z / (4 * checked**2)) / (1 + z * z / checked)
    return centre - half, centre + half


def error_note() -> str:
    """The share of passed outlines that are wrong by the owner's standard."""
    wrong, checked = OWNER_FOUND_WRONG
    low, high = wilson(wrong, checked)
    return (
        f"{wrong} of {checked} owner-checked helper passes wrong"
        f" (95% interval {low:.0%} to {high:.0%}; agatesim ADR 0075)"
    )


def _answers(path: Path) -> tuple[str, dict[str, dict[str, str]]]:
    data = json.loads(path.read_text("utf-8"))
    run = str(data.pop("_run"))
    return run, data


def gather(*rim_checks: Path) -> dict[str, list[dict[str, str | bool]]]:
    """Verdicts on the model's outlines in rim-check folders, keyed by photo.
    Owner answers replace the helpers' on the same outline; where two runs
    judged the same outline differently, the "no" is kept."""
    found: dict[str, dict[str, dict[str, str | bool]]] = {}
    judges = [p for d in rim_checks for p in sorted(d.glob("judge-*.json"))]
    owners = [p for d in rim_checks for p in sorted(d.glob("owner-*.json"))]
    for path in [*judges, *owners]:
        by = "owner" if path.name.startswith("owner-") else "helper"
        run, answers = _answers(path)
        results_path = path.parent / run / "results.json"
        if not results_path.exists():
            continue
        results = json.loads(results_path.read_text("utf-8"))
        if results.get("method") != "C":
            continue
        masks = np.load(path.parent / run / "masks.npz")
        for image_id, answer in answers.items():
            if results["results"].get(image_id, {}).get("reason", "") is not None:
                continue  # judged a photo without an outline: nothing to tie it to
            key = reduced_fingerprint(masks[image_id])
            verdict: dict[str, str | bool] = {
                "mask": key,
                "rim_in_frame": answer["rim_in_frame"] == "yes",
                "follows": answer["outline_follows"] == "yes",
                "by": by,
                "run": run,
            }
            held = found.setdefault(image_id, {})
            before = held.get(key)
            if before is not None and before["by"] == by:
                for name in ("rim_in_frame", "follows"):
                    verdict[name] = bool(before[name] and verdict[name])
            held[key] = verdict  # owner files come last, so the owner's answer wins
    return {k: sorted(v.values(), key=lambda d: str(d["mask"])) for k, v in sorted(found.items())}


def write(verdicts: Mapping[str, list[dict[str, str | bool]]], path: Path = VERDICTS_TOML) -> None:
    doc = {"meta": {"source": "agatesim ADR 0075", "photos": len(verdicts)}, "images": verdicts}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + tomli_w.dumps(doc), "utf-8")


def load(path: Path = VERDICTS_TOML) -> Verdicts:
    """Verdicts by photo, then by outline fingerprint."""
    if not path.exists():
        return {}
    images = tomllib.loads(path.read_text("utf-8"))["images"]
    return {
        image_id: {
            v["mask"]: Verdict(v["rim_in_frame"], v["follows"], v["by"], v["run"]) for v in rows
        }
        for image_id, rows in images.items()
    }


def verdict(verdicts: Verdicts, image_id: str, mask: NDArray[np.bool_]) -> Verdict | None:
    """The judgment of this very outline, or None if it was never judged."""
    return verdicts.get(image_id, {}).get(fingerprint(mask))


@dataclass
class Finder:
    """agatesim's rim-finder with its models loaded from agatesim's folder
    (installed as a library, agatesim would look inside this environment)."""

    agatesim: Path
    model: Sam | None = None
    cache: PatchCache | None = None

    def __post_init__(self) -> None:
        from agatesim.classify import MODEL_PATH
        from agatesim.maps import OnnxPatchEmbedder, PatchCache
        from agatesim.specimen.sam import SAM_DIR, Sam

        models = self.agatesim / "references" / "models"
        self.model = Sam(models / SAM_DIR.name)
        self.cache = PatchCache(OnnxPatchEmbedder(models / MODEL_PATH.name), models)

    def find(self, image_id: str, data: bytes) -> tuple[NDArray[np.uint8], Rim]:
        """The photo at `RIM_PX` and its rim, with the model passed explicitly."""
        from agatesim.specimen.photo import load_photo
        from agatesim.specimen.rim import find_rim

        assert self.cache is not None
        rgb = load_photo(data, RIM_PX).rgb
        # agatesim's library cache key (agatesim.maps._library_key): id and file size.
        patches = self.cache.get(f"{image_id}-{len(data)}", lambda: rgb)
        return rgb, find_rim(rgb, patches=patches, sam=self.model)


def unjudged(tags_toml: Path) -> list[str]:
    """Photos in a tag file whose rim outline was found but never judged."""
    images = tomllib.loads(tags_toml.read_text("utf-8"))["images"]
    return sorted(
        k for k, r in images.items() if r["rim"] == "found" and r["outline_check"] == "unknown"
    )


def save_run(finder: Finder, images: Mapping[str, Path], out: Path) -> None:
    """A rim-check run of the model's outlines, laid out as agatesim's
    `scripts/rim_check.py` lays one out: results.json, masks.npz (masks at
    `MASK_PX`) and a picture of each outline (green, no metadata) to judge."""
    from nodule.overlay import GREEN, edge, label

    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, object]] = {}
    masks: dict[str, NDArray[np.bool_]] = {}
    for image_id, path in images.items():
        rgb, rim = finder.find(image_id, path.read_bytes())
        results[image_id] = {"reason": rim.reason, "area": float(rim.mask.mean())}
        small = Image.fromarray(rim.mask)
        small.thumbnail((MASK_PX, MASK_PX), Image.Resampling.NEAREST)
        masks[image_id] = np.asarray(small, dtype=bool)
        pixels = rgb.copy()
        line = edge(rim.mask)
        for shift in ((0, 0), (0, 1), (1, 0), (1, 1)):  # a line 2 px wide
            pixels[np.roll(line, shift, axis=(0, 1))] = GREEN
        note = "outline" if rim.found else f"no outline: {rim.reason}"
        label(Image.fromarray(pixels), note).save(out / f"{image_id}.png")
    settings = {"rim_px": RIM_PX, "method": "C"}
    (out / "results.json").write_text(json.dumps({**settings, "results": results}, indent=1))
    np.savez_compressed(out / "masks.npz", **masks)  # pyright: ignore[reportArgumentType]


def main(argv: Sequence[str] | None = None) -> int:
    from agatesim.library import load_library

    from nodule.tags import TAGS_TOML, agatesim_dir, image_path

    parser = argparse.ArgumentParser(description="bring in the rim-outline verdicts")
    parser.add_argument("--out", type=Path, default=VERDICTS_TOML)
    parser.add_argument(
        "--draw-unjudged",
        metavar="NAME",
        help="save the tag file's unjudged outlines as run NAME in runs/rim-check, to be judged",
    )
    args = parser.parse_args(argv)
    agatesim = agatesim_dir()
    rim_check = agatesim / "runs" / "rim-check"
    if not rim_check.exists():
        print(f"no rim-check runs at {rim_check}; set NODULE_AGATESIM")
        return 1
    if args.draw_unjudged:
        library = load_library(agatesim / "references" / "library.toml")
        ids = unjudged(TAGS_TOML)
        out = OWN_RUNS / args.draw_unjudged
        save_run(Finder(agatesim), {k: image_path(agatesim, library.get(k)) for k in ids}, out)
        print(f"saved {len(ids)} unjudged outlines to {out}; answers go to judge-{out.name}.json")
        return 0
    verdicts = gather(rim_check, OWN_RUNS)
    write(verdicts, args.out)
    rows = [v for vs in verdicts.values() for v in vs]
    passed = sum(bool(v["rim_in_frame"] and v["follows"]) for v in rows)
    owner = sum(v["by"] == "owner" for v in rows)
    print(
        f"{len(rows)} judged outlines on {len(verdicts)} photos ({passed} passed,"
        f" {owner} answered by the owner) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
