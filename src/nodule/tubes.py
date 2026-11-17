"""Escape tubes (infiltration channels) as agatesim's wall channels: the wall bands thin
to nothing toward a line on the wall and close on it, Walger's form (agatesim ADR 0077;
work plan item 15; ADR 0018 s.18).

Walger: an onion-shaped section "a few mm" to 2-2.5 cm deep, the layers thinning "to
zero" toward a focus at the wall; none below cherry size, often one at fist size
[@walger2009, pp. 117, 125, 128] (*verified* as the source's words). The sources disagree
on how often tubes occur and on what made them (ADR 0018 s.18); appearance only, never
evidence for a mechanism (ADR 0002).

The owner's choices (2026-11-17): none on pebbles; about 1 in 6 of the other stones;
stack depth 3-10 mm. Lateral reach 8 mm and line length 10 mm are agatesim's defaults,
as on n2-slice, which the owner judged in ADR 0015 s.6. The depth is capped at half the
stone's chalcedony thickness: 15 mm stacks opened voids on three of four probe stones
(local `runs/tube-probe/`). All values `effective`.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

TUBE_CHANCE = 1 / 6
NO_TUBES = ("pebble",)  # Walger: none below cherry size
DEPTH_MM = (3.0, 10.0)  # log-uniform
LATERAL_MM = 8.0
LENGTH_MM = 10.0
DEPTH_CAP_SHARE = 0.5  # of the chalcedony's thickness
MIN_DEPTH_MM = 1.0  # below this, after the cap, the stone takes no tube


class Channel(BaseModel):
    """One wall channel (agatesim's settings)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    depth_mm: float = Field(gt=0, description="Thickness of the stack of bands that thin out.")
    lateral_mm: float = Field(default=LATERAL_MM, gt=0)
    length_mm: float = Field(default=LENGTH_MM, gt=0)

    def model_params(self) -> dict[str, object]:
        return {
            "channel_count": 1,
            "channel_depth_mm": self.depth_mm,
            "channel_lateral_mm": self.lateral_mm,
            "channel_length_mm": self.length_mm,
        }

    def summary(self) -> str:
        return f"tube {self.depth_mm:.1f} mm"


def draw_channel(rng: np.random.Generator, outline: str, chalcedony_mm: float) -> Channel | None:
    """One stone's tube, or none. Every stone draws the same numbers, so a later change
    of eligibility moves no other stone's draw."""
    u = rng.random(2)
    if outline in NO_TUBES or u[0] >= TUBE_CHANCE:
        return None
    lo, hi = DEPTH_MM
    depth = math.exp(math.log(lo) + float(u[1]) * (math.log(hi) - math.log(lo)))
    depth = min(depth, DEPTH_CAP_SHARE * chalcedony_mm)
    return Channel(depth_mm=depth) if depth >= MIN_DEPTH_MM else None
