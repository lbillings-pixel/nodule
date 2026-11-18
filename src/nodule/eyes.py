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
stone, but eyes showing few rings (coarse bands) quite rare. The rates are assumptions.
All values `effective`.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# Outlines that take eyes: the volcanic ones (vesicle, prolate: Pabian's amygdaloidal
# agates) and thunder eggs. Not on sediment stones (irregular) or pebbles.
EYE_OUTLINES = ("vesicle", "prolate", "thunder-egg")
# Chance on an eligible stone (no floor, crystals, plumes or moss: these hold the wall's
# first stage, `eligible`) whose eyes show at least MIN_RINGS rings. Set so that the
# expected share over sheets 0-59 is 1 vesicle or prolate stone in 10 and 1 thunder egg
# in 50 (local `runs/eye-probe/calibrate.py`; ADR 0018 s.18).
EYE_CHANCE = {"vesicle": 0.263, "prolate": 0.263, "thunder-egg": 0.035}
MIN_RINGS = 3.0  # radius / band width: fewer rings than this ...
FEW_RINGS_WEIGHT = 0.15  # ... and the chance is this share (owner: "quite rare")
RADIUS_MM = (2.0, 4.0)  # uniform: 4-8 mm across (Pabian, Fig. 38)
SPACING_PER_RADIUS = (2.2, 6.0)  # uniform: touching (2.2) to well apart (6)
RATE_VARIABILITY = 0.2  # spread of the domes' sizes within a stone
# How far the domes move the later bands inward where the centre forms, as a share of
# the radius, against spacing / radius. Measured on two vesicles at radii 2, 3, 4 mm
# (local `runs/eye-probe/advance.py`): the push grows with depth, from 0.3-1.0 R at 6 mm
# from the wall to 1.0-1.4 R at 15-18 mm, as the domes' tips come to set the fronts. The
# domes take R of the run, so the run changes by R minus this. Checked on the quartz-
# centre eyed stones of sheets 0-11 (`build_check.py`): quartz +2 % to +35 %, open
# centres 84-92 % of the same stones without eyes (0.6-1.0 left them 14-71 % smaller).
ADVANCE_AT = (2.2, 3.5, 6.0)
ADVANCE_SHARE = (1.2, 1.1, 1.0)


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

    def advance_mm(self) -> float:
        """How far the domes move the later bands inward (`ADVANCE_SHARE`)."""
        k = self.spacing_mm / self.radius_mm
        return float(np.interp(k, ADVANCE_AT, ADVANCE_SHARE)) * self.radius_mm

    def summary(self) -> str:
        return f"eyes {2 * self.radius_mm:.0f} mm"


def eligible(outline: str, floor: float, early: bool) -> bool:
    """Whether a stone may have eyes: an eye outline, no floor and no early stage
    (crystals, plumes or moss), which would hold the wall's first stage."""
    return outline in EYE_OUTLINES and not floor and not early


def draw_eyes(rng: np.random.Generator, outline: str, band_mm: float, ok: bool) -> Eyes | None:
    """One stone's eyes, or none. Every stone draws the same numbers, so a later change
    of eligibility moves no other stone's draw."""
    u = rng.random(3)
    radius = RADIUS_MM[0] + float(u[1]) * (RADIUS_MM[1] - RADIUS_MM[0])
    if not ok:
        return None
    chance = EYE_CHANCE[outline]
    if radius / band_mm < MIN_RINGS:
        chance *= FEW_RINGS_WEIGHT
    if u[0] >= chance:
        return None
    lo, hi = SPACING_PER_RADIUS
    return Eyes(radius_mm=radius, spacing_mm=radius * (lo + float(u[2]) * (hi - lo)))
