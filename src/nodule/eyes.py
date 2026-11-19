"""Eyes: banded domes on the wall that the later bands drape over, drawn as agatesim's
spherulitic stage (blue-noise sites on the wall, each growing a dome; agatesim ADR 0023,
0024, 0031). Work plan item 15; ADR 0018 s.18.

Pabian: "concentrically arranged bands that form a hemisphere with its flat surface
showing usually on the exterior of the agate nodule", most common in amygdaloidal
agates, very few in thunder eggs, none in sedimentary agates; on his Fig. 38 eyes are
4-8 mm across, neighbours touching [@pabian1994, pp. 21-22] (*plausible*, a collectors'
circular, sizes by eye). That an eye is an early wall spherulite is an inference, not a
source's claim: the dome is the form drawn, never evidence for a mechanism (ADR 0002).

The owner's choices (2026-11-18, after the probe in local `runs/eye-probe/`): about 1
vesicle stone in 10, about 1 thunder egg in 50, none on sediment stones or pebbles; size
4-8 mm across and spacing from touching to well apart, drawn per stone; any banded
stone, but eyes showing few rings (coarse bands) quite rare; floor stones too, the floor
settling first (later the same day). The rates are assumptions.
All values `effective`.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# Outlines that take eyes: the volcanic ones (vesicle, prolate: Pabian's amygdaloidal
# agates) and thunder eggs. Not on sediment stones (irregular) or pebbles.
EYE_OUTLINES = ("vesicle", "prolate", "thunder-egg")
# Chance on an eligible stone (no crystals, plumes or moss: these hold the wall's first
# stage, `eligible`) whose eyes show at least MIN_RINGS rings. Set so that the
# expected share over sheets 0-59 is 1 vesicle or prolate stone in 10 and 1 thunder egg
# in 50 (local `runs/eye-probe/calibrate.py`; ADR 0018 s.18).
EYE_CHANCE = {"vesicle": 0.307, "prolate": 0.307, "thunder-egg": 0.054}
# The radius is capped at this share of the stone's chalcedony thickness, as tube depths
# are (`nodule.tubes`): the domes push the bands up to 1.2 R inward, so a thin wall must
# keep chalcedony after them. Where the cap falls below the smallest radius (walls under
# 4 mm, some floor stones), the stone takes no eyes.
RADIUS_CAP_SHARE = 0.5
# Bands across the radius (radius / band width) below which an eye shows few rings: a
# face just under the skin (cut B, 0.4 R in) crosses only the outer 0.6 R of the dome,
# so it shows about 0.6 of them as rings. On the eyed stones of sheets 0-11, ratios 3.3-
# 4.9 showed 1-3 rings on cut B and 6.5-7 about 3-4 (by eye, `recipe_sheet.py`): below
# 6, an eye reads as a plain dome with a ring or two ...
MIN_RINGS = 6.0
FEW_RINGS_WEIGHT = 0.08  # ... and the chance is this share (owner: "quite rare")
RADIUS_MM = (2.0, 4.0)  # uniform: 4-8 mm across (Pabian, Fig. 38)
SPACING_PER_RADIUS = (2.2, 6.0)  # uniform: touching (2.2) to well apart (6)
RATE_VARIABILITY = 0.2  # spread of the domes' sizes within a stone
# How far the domes move the later bands inward where the centre forms, as a share of
# the radius, against spacing / radius. Measured on two vesicles at radii 2, 3, 4 mm
# (local `runs/eye-probe/advance.py`): the push grows with depth, from 0.3-1.0 R at 6 mm
# from the wall to 1.0-1.4 R at 15-18 mm, as the domes' tips come to set the fronts. The
# domes take R of the run, so the run changes by R minus this. Checked on the quartz-
# centre eyed stones of sheets 0-11 (`build_check.py`): centre width (cube root of the
# quartz and open centre's volume) 1.02-1.17 of the same stones without eyes, except a
# tiny centre (5-5, 0.2 % of the stone) at 1.6; a push of 0.6-1.0, measured near the
# wall, had left them clearly smaller. The push depends on where the domes fall, so this
# is a mean, not a per-stone value.
ADVANCE_AT = (2.2, 3.5, 6.0)
ADVANCE_SHARE = (1.2, 1.1, 1.0)
# A thunder egg's centre lies far from most of its wall (the star's arms, or a lens's
# long sides), so domes there push its centre less, and far less when they stand apart.
# Per spacing (ADVANCE_AT), the share that kept the centre as without eyes on six 3D
# eggs (2 winding, 2 jagged, 2 lens; quartz-filled and -lined), medians of 0.73, 0.57
# and 0.13 (lenses ran higher at 2.2 R: 0.75-0.93; local `runs/thunderegg-3d/
# eye_push_3d.py`; ADR 0018 s.23). The earlier single 0.35 was measured on one prism
# star (16-4), before thunder eggs were drawn in 3D.
THUNDER_EGG_ADVANCE_SHARE = (0.75, 0.55, 0.15)
# Above a floor the centre faces the floor's top, where few domes grow, so the push there
# is also smaller: about 0.45 R kept the centres of 20-2 and 23-6 (quartz-lined) as
# without eyes (0.35 R: 83 % and 81 % of their quartz; 0.7 R: 120 % and 430 %); at the
# vesicles' 1.0-1.2 R both opened wide and solid 13-7 a void (local
# `runs/eye-probe/floor_push.py`).
FLOOR_ADVANCE_SHARE = 0.45


class Eyes(BaseModel):
    """A stone's eyes (agatesim's spherulitic stage settings)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    radius_mm: float = Field(gt=0, description="Dome radius: the eye stage's growth.")
    spacing_mm: float = Field(gt=0, description="Mean spacing of the domes' sites.")
    rate_variability: float = Field(default=RATE_VARIABILITY, ge=0, lt=1)

    def model_params(self) -> dict[str, object]:
        return {
            "spherulite_spacing_mm": self.spacing_mm,
            "spherulite_rate_variability": self.rate_variability,
        }

    def advance_mm(self, outline: str, floor: float = 0.0) -> float:
        """How far the domes move the later bands inward where the centre forms
        (`ADVANCE_SHARE`; a thunder egg's `THUNDER_EGG_ADVANCE_SHARE`, a floor stone's
        `FLOOR_ADVANCE_SHARE`)."""
        k = self.spacing_mm / self.radius_mm
        if outline == "thunder-egg":
            return float(np.interp(k, ADVANCE_AT, THUNDER_EGG_ADVANCE_SHARE)) * self.radius_mm
        if floor:
            return FLOOR_ADVANCE_SHARE * self.radius_mm
        return float(np.interp(k, ADVANCE_AT, ADVANCE_SHARE)) * self.radius_mm

    def summary(self) -> str:
        return f"eyes {2 * self.radius_mm:.0f} mm"


def eligible(outline: str, early: bool) -> bool:
    """Whether a stone may have eyes: an eye outline and no early stage (crystals,
    plumes or moss), which would hold the wall's first stage. A floor is allowed: it
    settles first and the eyes grow after it, on the walls and the floor's top (the
    owner's choice, 2026-11-18)."""
    return outline in EYE_OUTLINES and not early


def draw_eyes(
    rng: np.random.Generator, outline: str, band_mm: float, ok: bool, chalcedony_mm: float
) -> Eyes | None:
    """One stone's eyes, or none. Every stone draws the same numbers, so a later change
    of eligibility moves no other stone's draw."""
    u = rng.random(3)
    radius = RADIUS_MM[0] + float(u[1]) * (RADIUS_MM[1] - RADIUS_MM[0])
    radius = min(radius, RADIUS_CAP_SHARE * chalcedony_mm)
    if not ok or radius < RADIUS_MM[0]:  # too thin a wall for a 4 mm eye: none
        return None
    chance = EYE_CHANCE[outline]
    if radius / band_mm < MIN_RINGS:
        chance *= FEW_RINGS_WEIGHT
    if u[0] >= chance:
        return None
    lo, hi = SPACING_PER_RADIUS
    return Eyes(radius_mm=radius, spacing_mm=radius * (lo + float(u[2]) * (hi - lo)))
