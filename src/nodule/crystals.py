"""Early crystals: blades and fine needles grown on the wall before the bands, which
then wrap them (work plan items 9 + 10; ADR 0018 s.15).

Each crystal stone runs agatesim's stick stage for a moment at the start of growth, in
the slot plumes and moss use (`Growth.with_plumes`): the stage lays no silica, and the
bands that follow wrap the crystals at the full rate of the wall (agatesim's fixed
rate; the owner kept it, 2026-11-16). At the end each crystal is left white (a remnant),
turned to clear silica (a pseudomorph) or dissolved (a hollow mould).

Grades (ADR 0002): early wall crystals "always overgrown" by a chalcedony crust are
*verified* in large Deccan cavities [@ottens2019, s. 4.2.3], with early zeolites under
1 mm (s. 4.2.7); that concentric agate bands drape them in banded nodules is only
*plausible* (the paper describes crusts, not nodules). Silica pseudomorphs and moulds
after such crystals are frequent in agate [@goetze2020, p. 27]. The white colour of a
remnant is an assumption (no source gives it, so it is set by the owner's eye); sizes
on the blades, rates and shares are assumptions.

The owner's choices (look sheet 1, runs/crystal-probe, 2026-11-16): all six looks
shown were judged realistic, so a stone draws one of them, with a little spread in
size. All values `effective`, chosen by eye:
- small blades: spacing 2.5 mm, length 1.5 mm, elongation 6, scatter 30 deg; white,
  clear or hollow;
- fine needles: 1.0 mm, 0.9 mm, 12, 45 deg; white or hollow (clear not shown);
- large blades: 4.0 mm, 3.0 mm, 8, 30 deg; white (the only fill shown).

Crystals may share a stone with plumes or moss (the owner, 2026-11-16: "some combos"):
the stick stage comes first and the dendrite stage follows, seeding on the wall and the
crystals both, so plumes or moss may root on crystals (an assumption, by eye; no
source read gives it). Appearance only: never evidence for a formation mechanism
(ADR 0002).
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from agatesim.fields import Fields, Phase
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from nodule.plumes import STAGE, thinned

Habit = Literal["small blades", "fine needles", "large blades"]
CrystalFill = Literal["white", "clear", "hollow"]

# Of a sheet's stones, among those free: 1 in 9 (the owner, 2026-11-16: "1 in 8-10"),
# a thinning of the 1 in 6 of look sheet 2 (`POOL_SHARE`), so kept stones stand.
CRYSTAL_SHARE = 1 / 9
POOL_SHARE = 1 / 6
NO_CRYSTALS = ("pebble",)  # small and nearly unbanded (ADR 0018 s.7)
# Combinations (the owner, 2026-11-16 evening): 1 in 100-200 crystal stones also carry
# plumes or moss. Only stones without a floor and not thunder eggs qualify for now (about
# half of crystal stones: 8 of 17 on sheets 0-11), so each qualifying stone takes one
# with this chance: about 1 in 150 crystal stones. Plumes or moss, half each (assumption).
COMBO_CHANCE = 1 / 75
COMBO_PLUMES = 0.5
NO_COMBOS = ("thunder-egg",)

# Per habit: spacing (mm), length (mm), elongation, axis scatter (deg); the probe's.
HABITS: dict[str, tuple[float, float, float, float]] = {
    "small blades": (2.5, 1.5, 6.0, 30.0),
    "fine needles": (1.0, 0.9, 12.0, 45.0),
    "large blades": (4.0, 3.0, 8.0, 30.0),
}
HABIT_SHARES: dict[str, float] = {  # assumption: the large blades are the boldest look
    "small blades": 0.4,
    "fine needles": 0.4,
    "large blades": 0.2,
}
FILLS: dict[str, dict[str, float]] = {  # only the fills the owner saw; shares assumed
    "small blades": {"white": 0.5, "clear": 0.25, "hollow": 0.25},
    "fine needles": {"white": 0.5, "hollow": 0.5},
    "large blades": {"white": 1.0},
}
SIZE_SPREAD = (0.8, 1.25)  # log-uniform factor on spacing and length (assumption)
# Crystals reach at most this share of the chalcedony's thickness, so large blades
# stay off the centre of a small stone or a wide centre (thunder eggs: NEEDLES_ONLY).
MOST_REACH_SHARE = 0.25
# Thunder eggs take fine needles only (the owner's fallback, 2026-11-16): blades lump
# across the narrow arms (half-width 1.1-3.1 mm, median 1.7, over 48 stones), and a
# length cap by arm width shrinks every habit to 0.2-0.7 mm, specks on the face at
# 8-10 x the build time (runs/crystal-probe/thunderegg_cap.py; ADR 0018 s.15).
NEEDLES_ONLY = ("thunder-egg",)

# A white remnant: milky chalcedony's stand-in at this translucency, a little varied
# per crystal (the probe's chalky white, 210-226 grey; `effective`, by eye).
WHITE_TRANSLUCENCY = (0.04, 0.10)


class Crystals(BaseModel):
    """One stone's early crystals (agatesim stick settings)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    habit: Habit
    fill: CrystalFill
    spacing_mm: float = Field(gt=0, description="Mean spacing of crystal starts.")
    length_mm: float = Field(gt=0, description="Length along the crystal's axis.")
    elongation: float = Field(ge=2, le=60)
    scatter_deg: float = Field(ge=0, le=90)

    def model_params(self) -> dict[str, object]:
        return {
            "stick_spacing_mm": self.spacing_mm,
            "stick_length_mm": self.length_mm,
            "stick_elongation": self.elongation,
            "stick_axis_scatter_deg": self.scatter_deg,
            "stick_fill": "hollow" if self.fill == "hollow" else "quartz",
        }

    def capped(self, chalcedony_mm: float) -> Crystals:
        """These crystals, shrunk (spacing with length, so the look stays) to reach at
        most MOST_REACH_SHARE of the chalcedony's thickness."""
        most = MOST_REACH_SHARE * chalcedony_mm
        if self.length_mm <= most:
            return self
        k = most / self.length_mm
        return self.model_copy(
            update={"length_mm": self.length_mm * k, "spacing_mm": self.spacing_mm * k}
        )

    def as_needles(self) -> Crystals:
        """The same draw as fine needles: the same size factor, and white unless the
        draw was hollow (clear needles were not on the owner's sheet)."""
        if self.habit == "fine needles":
            return self
        k = self.length_mm / HABITS[self.habit][1]
        spacing, length, elongation, scatter = HABITS["fine needles"]
        return Crystals(
            habit="fine needles",
            fill="hollow" if self.fill == "hollow" else "white",
            spacing_mm=spacing * k,
            length_mm=length * k,
            elongation=elongation,
            scatter_deg=scatter,
        )

    def summary(self) -> str:
        return f"{self.fill} {self.habit} {self.length_mm:.1f} mm"


def _pick(u: float, shares: dict[str, float]) -> str:
    total, edge = sum(shares.values()), 0.0
    for name, share in shares.items():
        edge += share / total
        if u < edge:
            return name
    return list(shares)[-1]


def crystal_plan(rng: np.random.Generator, outlines: list[str], taken: set[int]) -> set[int]:
    """Which stones of a sheet carry early crystals: never a pebble, nor a stone with
    plumes or moss."""
    ok = [i for i, o in enumerate(outlines) if o not in NO_CRYSTALS and i not in taken]
    return thinned(rng, ok, POOL_SHARE, CRYSTAL_SHARE, len(outlines))[1]


def combo_plan(
    rng: np.random.Generator, eligible: list[bool], chance: float = COMBO_CHANCE
) -> dict[int, Literal["plumes", "moss"]]:
    """Which crystal stones of a sheet also carry plumes or moss, and which. Every stone
    draws its numbers, eligible or not, so widening the pool later moves no stone's
    draw."""
    u, kind = rng.random(len(eligible)), rng.random(len(eligible))
    return {
        i: "plumes" if kind[i] < COMBO_PLUMES else "moss"
        for i, ok in enumerate(eligible)
        if ok and u[i] < chance
    }


def draw_crystals(rng: np.random.Generator) -> Crystals:
    u = rng.random(3)
    habit = _pick(float(u[0]), HABIT_SHARES)
    fill = _pick(float(u[1]), FILLS[habit])
    spacing, length, elongation, scatter = HABITS[habit]
    lo, hi = SIZE_SPREAD
    k = float(np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * float(u[2])))
    return Crystals(
        habit=cast(Habit, habit),
        fill=cast(CrystalFill, fill),
        spacing_mm=spacing * k,
        length_mm=length * k,
        elongation=elongation,
        scatter_deg=scatter,
    )


def stage_mask(fields: Fields, inside: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Crystals laid down in the early stage (white or clear): growth time within the
    stage's span (always the first early stage), not an inclusion. Only in a crystal
    stone are these the crystals (elsewhere plumes, moss or the first bands lie there).
    Hollow moulds carry no growth time and are not found here; the same stone drawn
    clear has its crystals at the same points (the fill only changes the phase)."""
    t = np.nan_to_num(fields.t_dep, nan=np.inf)
    return inside & (t <= STAGE + 1e-6) & (fields.phase != Phase.INCLUSION)


def white_translucency(grain: NDArray[np.int32]) -> NDArray[np.float32]:
    """Each white crystal's own translucency, by its grain number."""
    shade = ((grain.astype(np.int64) * 7919 + 13) % 1000) / 1000.0
    lo, hi = WHITE_TRANSLUCENCY
    return (lo + (hi - lo) * shade).astype(np.float32)
