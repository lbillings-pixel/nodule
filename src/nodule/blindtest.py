"""The owner's blind test (ADR 0008 section 5) and its small trial (section 5a).

Two images side by side, one a photo and one a synthetic cut made to look like
that photo; the owner picks the synthetic one. No feedback and no score until
the end. The steps, each its own command:

- `render OUT`: renders the cuts and saves each one finished as `measure` sees
  it (the photo's blur, and only the noise the render still lacks). GPU; a rerun
  skips the cuts already saved.
- `pairs OUT`: both images of a pair get the same square crop size inside the
  face, each its own position, flip and quarter-turn, against recognising a
  photo from memory. Writes neutral file names and, apart from them, the key.
- `serve OUT`, then `score OUT`: the page (this PC only), then the result.

Library photos only, never the owner's specimens. OUT goes under `runs/`
(gitignored): photos and crops never enter the repository or leave the PC.
PNGs are written from pixel arrays, so they carry no metadata. Appearance only:
never evidence for a formation mechanism (ADR 0002).
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import webbrowser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from nodule import check, rims
from nodule._scipy import distance_inside
from nodule.tags import agatesim_dir, image_path

PAGE = Path(__file__).with_name("blindtest.html")
HOST = "127.0.0.1"
PORT = 8768
SIDES = ("a", "b")

MIN_CROP_PX = 128  # below this a crop shows too few bands to judge (assumption)
LOWEST_SHARE = 0.6  # crop sizes are drawn from this share of the largest common square up
MARGIN_SHARE = 0.01  # of the photo's long side, kept off a found face's edge
SEED_SPLIT = 2026  # the compare's calibration split (as its run scripts)

Rgb = NDArray[np.uint8]
Mask = NDArray[np.bool_]


@dataclass(frozen=True)
class Plan:
    """How many cuts of each run, and how many pairs each cut gives."""

    stone_cuts: int
    stone_crops: int
    catch_cuts: Mapping[str, int]
    obvious_from: int | None  # correct main pairs that say "spotted at a glance"
    seed: int


# Section 5a. Its own seed, so its cuts stay apart from the full test's.
TRIAL = Plan(8, 2, {"F1": 2, "F3": 2}, obvious_from=14, seed=1)


@dataclass(frozen=True)
class Cut:
    """A finished synthetic cut and the photo it was made to look like."""

    run: str
    name: str
    photo: str
    rgb: Rgb
    inside: Mask
    matched_noise: bool | None = None  # the render's noise at or below the photo's


def binomial_tail(correct: int, trials: int) -> float:
    """P(at least `correct` of `trials` by coin flip): the exact one-sided test."""
    return sum(math.comb(trials, k) for k in range(correct, trials + 1)) / 2**trials


def _fits(mask: Mask, size: int) -> Mask:
    """Top-left corners where a `size` square lies wholly inside `mask`."""
    h, w = mask.shape
    if size > min(h, w):
        return np.zeros((0, 0), bool)
    area = np.zeros((h + 1, w + 1), np.int64)
    area[1:, 1:] = mask.astype(np.int64).cumsum(0).cumsum(1)
    inside = area[size:, size:] - area[:-size, size:] - area[size:, :-size] + area[:-size, :-size]
    return inside == size * size


def largest_square(mask: Mask) -> int:
    """The side of the largest square wholly inside `mask` (0 if it is empty)."""
    low, high = 0, min(mask.shape)
    while low < high:
        mid = (low + high + 1) // 2
        if _fits(mask, mid).any():
            low = mid
        else:
            high = mid - 1
    return low


def crop(rgb: Rgb, mask: Mask, size: int, rng: np.random.Generator) -> Rgb:
    """A random `size` square inside `mask`, randomly flipped and quarter-turned."""
    rows, cols = np.nonzero(_fits(mask, size))
    if rows.size == 0:
        raise ValueError(f"no {size} px square fits inside the face")
    at = int(rng.integers(rows.size))
    r, c = int(rows[at]), int(cols[at])
    out = rgb[r : r + size, c : c + size]
    if rng.integers(2):
        out = out[:, ::-1]
    return np.rot90(out, int(rng.integers(4))).copy()


def crop_size(top: int, rng: np.random.Generator) -> int:
    """A crop size for two faces whose largest common square is `top`."""
    if top < MIN_CROP_PX:
        raise ValueError(f"the faces share only a {top} px square")
    return int(rng.integers(max(MIN_CROP_PX, round(LOWEST_SHARE * top)), top + 1))


def _png(rgb: Rgb, path: Path) -> None:
    Image.fromarray(rgb).save(path, format="PNG")  # from pixels: no metadata


# Rendering (GPU) and the photos' faces.


def half_a_looks() -> list[Any]:
    """The compare's looks: calibration half A of the reflected photos."""
    from nodule import compare
    from nodule.descriptors import library_looks, stratum_looks

    ids = np.array(compare.library_table("reflected").ids)
    perm = np.random.default_rng(SEED_SPLIT).permutation(len(ids))
    half = set(ids[perm[: len(ids) // 2]])
    return [lk for lk in stratum_looks(library_looks()[0], "reflected") if lk.photo in half]


def pulled_in(mask: Mask, margin: int) -> Mask:
    """`mask` less everything within `margin` px of its edge."""
    return distance_inside(np.pad(mask, 1))[1:-1, 1:-1] > margin


class Faces:
    """Library photos at full resolution (the size their `Look` was measured
    at) with their faces there: the rim outline where it was judged "yes", or
    else agatesim's region, as for the photo's tags, kept off the edge."""

    def __init__(self) -> None:
        from agatesim.library import load_library

        self.agatesim = agatesim_dir()
        self.library = load_library(self.agatesim / "references" / "library.toml")
        self.finder = rims.Finder(self.agatesim)
        self.verdicts = rims.load()

    def __call__(self, image_id: str) -> tuple[Rgb, Mask]:
        from agatesim.shapes import WORKING_PX
        from agatesim.specimen.photo import load_photo
        from agatesim.specimen.segment import segment

        data = image_path(self.agatesim, self.library.get(image_id)).read_bytes()
        _, rim = self.finder.find(image_id, data)
        judged = rims.verdict(self.verdicts, image_id, rim.mask) if rim.found else None
        passed = judged is not None and judged.passed
        small = rim.mask if passed else segment(load_photo(data, WORKING_PX).rgb).mask
        rgb = load_photo(data, 10**6).rgb
        h, w = rgb.shape[:2]
        full = Image.fromarray(small).resize((w, h), Image.Resampling.NEAREST)
        margin = max(1, round(MARGIN_SHARE * max(h, w)))
        return rgb, pulled_in(np.asarray(full, dtype=bool), margin)


def finish(pc: Any, look: Any, rng: np.random.Generator) -> tuple[Rgb, float] | None:
    """The rendered cut as `measure` sees it: the photo's blur, then only the
    noise it lacks. None where the blur cannot be found (no bands to match)."""
    from agatesim.cut_shapes import measure_cut, photo_like
    from agatesim.fields import Phase

    from nodule.descriptors import noise_to_add

    shape_rng, noise_rng = rng.spawn(2)
    lacking = noise_to_add(look, pc.rendered)
    quartz = pc.cut.fields.phase == Phase.QUARTZ
    sigma = measure_cut(pc.rgb, pc.cut.inside, quartz, lacking, shape_rng).get("sigma_px")
    if not isinstance(sigma, int | float):
        return None
    return photo_like(pc.rgb, float(sigma), lacking.noise, noise_rng), float(sigma)


def render_cuts(out: Path, plan: Plan, stop: int | None = None, **kw: Any) -> None:
    """Renders the plan's cuts into `out/cuts`, one file per usable cut, and logs
    each in `out/cuts.jsonl`. A plane without bands, or too small to crop, is
    logged and passed over: the next is drawn, up to twice the number wanted.
    `stop` ends this session after that many new renders; a rerun goes on with
    the same planes. `kw` goes to the rendering (as `most_spp`)."""
    from agatesim import progress
    from agatesim.volume.slicing import random_cuts

    from nodule.descriptors import MOST_PX, PROBE_PX, photo_cut, rendering_for
    from nodule.specimen import Specimen, load_preset

    (out / "cuts").mkdir(parents=True, exist_ok=True)
    log = out / "cuts.jsonl"
    rows = [json.loads(s) for s in log.read_text("utf-8").splitlines()] if log.exists() else []
    done = {row["name"]: row for row in rows}
    t0 = time.perf_counter()
    spec = Specimen.build(load_preset("n2-slice"))
    faces = Faces()
    looks = [lk for lk in half_a_looks() if largest_square(faces(lk.photo)[1]) >= MIN_CROP_PX]
    print(f"{len(looks)} usable photos, {time.perf_counter() - t0:.0f} s to prepare", flush=True)
    wanted = {"stone": plan.stone_cuts, **plan.catch_cuts}
    total = sum(wanted.values())
    made = fresh = 0
    streams = np.random.default_rng(plan.seed).spawn(len(wanted))
    for run, stream in zip(wanted, streams, strict=True):
        rendering = None
        cut_rng, look_rng, finish_rng, render_rng = stream.spawn(4)
        usable = 0
        planes = random_cuts(spec.volume.solid, 2 * wanted[run], cut_rng, PROBE_PX)
        for k, where in enumerate(planes):
            if usable == wanted[run]:
                break
            look = looks[int(look_rng.integers(len(looks)))]
            one_render, one_finish = render_rng.spawn(1)[0], finish_rng.spawn(1)[0]
            name = f"{run}-{k:03d}"
            if name in done:
                usable += bool(done[name]["usable"])
                made += bool(done[name]["usable"])
                continue
            if stop is not None and fresh >= stop:
                print(f"stopped after {fresh} new renders", flush=True)
                return
            progress.report(made / total, f"{run} cut {usable + 1} of {wanted[run]}")
            if rendering is None:  # the stone's record is read only if a cut needs it
                rendering = rendering_for(spec, run, "reflected", np.random.default_rng(0), **kw)
            t1 = time.perf_counter()
            pc = photo_cut(spec, spec.plane(where), look, MOST_PX, rendering, one_render)
            finished = finish(pc, look, one_finish)
            ok = finished is not None and largest_square(pc.cut.inside) >= MIN_CROP_PX
            row: dict[str, Any] = {"name": name, "run": run, "photo": look.photo, "usable": ok}
            row |= {"pixels": pc.pixels, "seconds": round(time.perf_counter() - t1, 1)}
            r = pc.rendered
            if r is not None:
                row |= {"spp": r.spp, "blurred_noise": r.blurred_noise,
                        "photo_noise": look.noise, "matched_noise": r.matched_noise}  # fmt: skip
            if finished is not None:
                row["sigma_px"] = round(finished[1], 3)
                if ok:
                    np.savez_compressed(out / "cuts" / f"{name}.npz",
                                        rgb=finished[0], inside=pc.cut.inside)  # fmt: skip
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            usable += ok
            made += ok
            fresh += 1
            print(json.dumps(row), flush=True)
    progress.report(1.0, "cuts rendered")


def load_cuts(out: Path) -> list[Cut]:
    found: list[Cut] = []
    for line in (out / "cuts.jsonl").read_text("utf-8").splitlines():
        row = json.loads(line)
        if row["usable"]:
            with np.load(out / "cuts" / f"{row['name']}.npz") as saved:
                found.append(Cut(row["run"], row["name"], row["photo"], saved["rgb"],
                                 saved["inside"], row.get("matched_noise")))  # fmt: skip
    return found


# Pairs and the key.


def build_pairs(
    cuts: Sequence[Cut],
    photos: Mapping[str, tuple[Rgb, Mask]],
    plan: Plan,
    out: Path,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Writes the pairs into `out/pairs` and returns the key, in showing order.
    Stone cuts give `plan.stone_crops` pairs and catch cuts one. Sides are
    balanced within the main and within the catch pairs."""
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    jobs = [c for c in cuts for _ in range(plan.stone_crops if c.run == "stone" else 1)]
    order_rng, side_rng, crop_rng = rng.spawn(3)
    order = [int(i) for i in order_rng.permutation(len(jobs))]
    side: dict[int, int] = {}
    for group in ([i for i in order if jobs[i].run == "stone"],
                  [i for i in order if jobs[i].run != "stone"]):  # fmt: skip
        sides = side_rng.permutation([k % 2 for k in range(len(group))])
        side |= {i: int(s) for i, s in zip(group, sides, strict=True)}
    key: list[dict[str, Any]] = []
    for n, i in enumerate(order, start=1):
        c = jobs[i]
        photo_rgb, face = photos[c.photo]
        size = crop_size(min(largest_square(c.inside), largest_square(face)), crop_rng)
        trial = f"t{n:03d}"
        synthetic, real = SIDES[side[i]], SIDES[1 - side[i]]
        _png(crop(c.rgb, c.inside, size, crop_rng), out / "pairs" / f"{trial}-{synthetic}.png")
        _png(crop(photo_rgb, face, size, crop_rng), out / "pairs" / f"{trial}-{real}.png")
        key.append({"trial": trial, "synthetic": synthetic, "run": c.run, "cut": c.name,
                    "photo": c.photo, "size": size,
                    "matched_noise": c.matched_noise})  # fmt: skip
    (out / "key.json").write_text(json.dumps(key, indent=1), "utf-8")
    return key


# The page.


class BlindTest:
    """The trials and the owner's answers. The page is given the trial numbers
    only: the key stays on disk and is not read while the page runs."""

    page = PAGE

    def __init__(self, out: Path) -> None:
        self.out = out
        self.answers_path = out / "answers.json"
        names = sorted(p.name for p in (out / "pairs").glob("t*-a.png"))
        self.trials = [n.removesuffix("-a.png") for n in names]
        self.answers: dict[str, dict[str, Any]] = (
            json.loads(self.answers_path.read_text("utf-8")) if self.answers_path.exists() else {}
        )
        self.lock = threading.Lock()

    def data(self) -> dict[str, object]:
        return {"trials": self.trials, "answered": sorted(self.answers)}

    def picture(self, name: str) -> Path | None:
        trial, _, side = name.partition("-")
        if trial in self.trials and side in SIDES:
            return self.out / "pairs" / f"{name}.png"
        return None

    def save(self, image_id: str, values: dict[str, Any]) -> list[str]:
        if image_id not in self.trials:
            return [f"{image_id!r} is not a trial"]
        pick, note, seconds = values.get("pick"), values.get("note", ""), values.get("seconds")
        if pick not in SIDES:
            return [f"pick: {pick!r} is not one of {SIDES}"]
        if not isinstance(note, str):
            return ["note: not text"]
        if isinstance(seconds, bool) or not isinstance(seconds, int | float):
            return ["seconds: not a number"]
        with self.lock:
            if image_id in self.answers:
                return [f"{image_id} is already answered"]
            self.answers[image_id] = {"pick": pick, "note": note, "seconds": round(seconds, 1)}
            self.answers_path.write_text(json.dumps(self.answers, indent=1), "utf-8")
        return []


# The result.


@dataclass(frozen=True)
class Score:
    main_correct: int
    main_trials: int
    catch_correct: int
    catch_trials: int
    p_value: float  # exact one-sided, treating pairs as independent
    interval: tuple[float, float]  # 95 %, on the main pairs
    first_half: tuple[int, int]  # correct, trials: main pairs in showing order
    second_half: tuple[int, int]
    by_run: dict[str, tuple[int, int]]
    by_noise: dict[str, tuple[int, int]]  # main pairs: was the render's noise matched?
    median_seconds: float
    obvious: bool | None  # the plan's "spotted at a glance", if it has one


def score(
    key: Sequence[Mapping[str, Any]], answers: Mapping[str, Mapping[str, Any]], plan: Plan
) -> Score:
    """The result, refused until every pair is answered (no score until the end)."""
    missing = [k["trial"] for k in key if k["trial"] not in answers]
    if missing:
        raise ValueError(f"not scored until every pair is answered; {len(missing)} left")
    hits = [(k, answers[k["trial"]]["pick"] == k["synthetic"]) for k in key]
    main = [hit for k, hit in hits if k["run"] == "stone"]
    catch = [hit for k, hit in hits if k["run"] != "stone"]
    by_run: dict[str, tuple[int, int]] = {}
    for k, hit in hits:
        right, of = by_run.get(k["run"], (0, 0))
        by_run[k["run"]] = (right + hit, of + 1)
    by_noise: dict[str, tuple[int, int]] = {}
    for k, hit in hits:
        if k["run"] == "stone":
            matched = k.get("matched_noise")
            group = "unknown" if matched is None else "matched" if matched else "noisier"
            right, of = by_noise.get(group, (0, 0))
            by_noise[group] = (right + hit, of + 1)
    mid = len(main) // 2
    wrong_low, wrong_high = rims.wilson(len(main) - sum(main), len(main))
    return Score(
        sum(main),
        len(main),
        sum(catch),
        len(catch),
        binomial_tail(sum(main), len(main)),
        (1.0 - wrong_high, 1.0 - wrong_low),
        (sum(main[:mid]), mid),
        (sum(main[mid:]), len(main) - mid),
        by_run,
        by_noise,
        float(np.median([answers[k["trial"]]["seconds"] for k in key])),
        None if plan.obvious_from is None else sum(main) >= plan.obvious_from,
    )


def report(s: Score, notes: Sequence[str]) -> str:
    low, high = s.interval
    lines = [
        f"main pairs: {s.main_correct} of {s.main_trials} correct (chance {s.main_trials // 2}); "
        f"one-sided p = {s.p_value:.4f}, treating pairs as independent; "
        f"95 % interval {100 * low:.0f} to {100 * high:.0f} %",
        f"catch pairs: {s.catch_correct} of {s.catch_trials} correct",
        f"first half {s.first_half[0]}/{s.first_half[1]}, "
        f"second half {s.second_half[0]}/{s.second_half[1]}",
        "by run: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(s.by_run.items())),
        "main pairs by render noise: "
        + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(s.by_noise.items())),
        f"median {s.median_seconds:.0f} s per pair",
    ]
    if s.obvious is not None:
        lines.append(f"spotted at a glance by the trial's rule: {'yes' if s.obvious else 'no'}")
    return "\n".join([*lines, *(f"note: {n}" for n in notes)])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="the owner's blind test (ADR 0008 section 5)")
    parser.add_argument("step", choices=("render", "pairs", "serve", "score"))
    parser.add_argument("out", type=Path, help="a folder under runs/")
    parser.add_argument("--stop", type=int, help="render: end after this many new renders")
    parser.add_argument("--most-spp", type=int, help="render: the rendering's cap on samples")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-open", action="store_true", help="serve: do not open the browser")
    args = parser.parse_args(argv)
    plan = TRIAL
    if args.step == "render":
        kw = {} if args.most_spp is None else {"most_spp": args.most_spp}
        render_cuts(args.out, plan, args.stop, **kw)
    elif args.step == "pairs":
        cuts = load_cuts(args.out)
        faces = Faces()
        photos = {photo: faces(photo) for photo in {c.photo for c in cuts}}
        rng = np.random.default_rng([plan.seed, 1])
        print(f"{len(build_pairs(cuts, photos, plan, args.out, rng))} pairs in {args.out}")
    elif args.step == "serve":
        test = BlindTest(args.out)
        server = check.ThreadingHTTPServer((HOST, args.port), check.handler(test))
        url = f"http://{HOST}:{args.port}/"
        print(f"blind test at {url} ({len(test.trials)} pairs); Ctrl+C to stop")
        if not args.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    else:
        key = json.loads((args.out / "key.json").read_text("utf-8"))
        answers = json.loads((args.out / "answers.json").read_text("utf-8"))
        notes = [f"{k['trial']}: {answers[k['trial']]['note']}" for k in key
                 if answers.get(k["trial"], {}).get("note")]  # fmt: skip
        text = report(score(key, answers, plan), notes)
        print(text)
        (args.out / "score.txt").write_text(text + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
