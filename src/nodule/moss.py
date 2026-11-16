"""Moss: agatesim's branched inclusions grown downward before the bands (ADR 0018 s.14).

Moss uses the same dendrite stage as plumes (`nodule.plumes`), tilted to grow down from
the roof and walls (agatesim ADR 0079), each strand inside a clear sheath that keeps
faint bands (agatesim ADR 0085), and sized to the lobe or pocket it grows in (agatesim
ADR 0086). A stone carries plumes or moss, never both: agatesim runs one dendrite stage.

The owner's choices (2026-11-15, the moss probe and look sheets 1-4, work plan item 8);
all values `effective`, chosen by eye:
- looks: C (short, patchy tufts) on quartz-lined, quartz-filled or open centres,
  growing toward E as the centre shrinks, with plain C kept on 1 stone in 5; E (long
  drapes) by default and G (dense, bulky) on 1 in 3 solid stones;
- moss is not only a "moss stone": short moss sits in ordinary banded agates too;
- colour mostly green, some yellow-brown, about 1 in 10 each red and black (the probe
  plan, ADR 0018 s.13; shares are assumptions);
- a clear sheath, keeping 40-60 % of the band colour ("good");
- reach a share of the local depth, with lobe contrast 2 (sheet 4, "let's do contrast 2").

`MOSS_SHARE` is an assumption while the look is honed (work plan item 11 then lowers
plumes to 1 in 8-10). Appearance only: never evidence for a formation mechanism
(ADR 0002).
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from nodule.plumes import CELLS_M

MossColour = Literal["green", "yellow-brown", "red", "black"]

MOSS_SHARE = 1 / 6  # of a sheet's stones, among those without plumes (assumption)
NO_MOSS = ("pebble",)  # small and nearly unbanded (ADR 0018 s.7)

# Per look: spacing (mm), amount, density, sticking, upward bias, start height. From the
# probe (runs/moss-probe, ADR 0018 s.14).
LOOKS: dict[str, tuple[float, float, float, float, float, float]] = {
    "C": (3.0, 0.03, 0.08, 0.3, 0.9, 1.0),
    "E": (4.0, 0.06, 0.04, 0.5, 0.9, 0.6),
    "G": (3.0, 0.10, 0.10, 0.5, 0.8, 1.0),
}
G_SHARE = 1 / 3  # of solid moss stones
PLAIN_C_SHARE = 1 / 5  # of moss stones with a quartz or open centre
# Short moss grows toward E as the centre shrinks: k = clip((C0 - width) / CW, 0, 1).
C_TO_E = (0.5, 0.4)
C_SPACING_MM = (3.0, 3.0)  # spacing 3 x 3^u mm: patchy, not an even border
C_START = (0.35, 0.65)  # start height 0.35 + 0.65 u: upper part of the stone only

COLOUR_SHARES: dict[str, float] = {"green": 0.7, "yellow-brown": 0.1, "red": 0.1, "black": 0.1}
TINTS: dict[str, float] = {"green": 0.03, "yellow-brown": 0.3, "red": 0.5, "black": 1.0}
TINT_CHANGE = 0.05
TINT_SPREAD = 0.05

SHEATH_SOLID_MM = (0.5, 1.5)  # 0.5 + 1.5 u
SHEATH_SHORT_MM = (0.3, 0.7, 0.5)  # 0.3 + 0.7 u + 0.5 k
CLEARING = (0.4, 0.2)  # 0.4 + 0.2 u: 40-60 % of band colour kept (the owner: "good")
WRAP_RATE = (0.15, 0.15, 1.5)  # 0.15 + 0.15 u^1.5, as plumes

# Reach (agatesim ADR 0086): a share of each lobe's or pocket's own depth.
SOLID_REACH_SHARE = 0.9
SHORT_REACH_SHARE = 0.6  # times (1 - centre width)
REACH_CONTRAST = 2.0  # the owner's pick, 2026-11-15
# But not in thunder eggs: their arms are narrow, so at 2 moss left them almost bare and
# crowded the centre ("very unphysical/artificial", the owner, 2026-11-16). At 0 each arm
# keeps moss in proportion to its own width.
THUNDER_EGG_CONTRAST = 0.0
# Share of the way from the wall to the centre (as plumes' CENTRE_SHARE).
SOLID_CENTRE_SHARE = 1.0
SHORT_CENTRE_SHARE = 2 / 3


class Moss(BaseModel):
    """One stone's moss (agatesim dendrite settings, ADR 0079, 0085, 0086)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    look: str = Field(description="E, G, or C>k (short, k toward E).")
    colour: MossColour
    tint: float = Field(ge=0, le=1, description="Base tint, agatesim's scale.")
    spacing_mm: float = Field(gt=0)
    amount: float = Field(gt=0, le=1)
    density: float = Field(gt=0, le=1)
    sticking: float = Field(gt=0, le=1)
    upward: float = Field(ge=0, le=1, description="Bias along the growth direction.")
    start_height: float = Field(ge=0, le=1)
    halo_mm: float = Field(gt=0, description="Clear sheath round each strand.")
    clearing: float = Field(ge=0, le=1, description="Share of band colour cleared.")
    wrap_rate: float = Field(gt=0, le=1, description="Band growth round moss / wall's.")
    reach_share: float = Field(gt=0, le=1)
    reach_floor: float = Field(ge=0, lt=1)
    centre_share: float = Field(gt=0, le=1)
    reach_contrast: float = Field(default=REACH_CONTRAST, ge=0, le=4)

    def model_params(self) -> dict[str, object]:
        return {
            "dendrite_spacing_mm": self.spacing_mm,
            "dendrite_amount": self.amount,
            "dendrite_density": self.density,
            "dendrite_sticking": self.sticking,
            "dendrite_upward": self.upward,
            "dendrite_start_height": self.start_height,
            "dendrite_cells_3d": CELLS_M,
            "dendrite_wrap_rate": self.wrap_rate,
            "dendrite_tint": self.tint,
            "dendrite_tint_change": TINT_CHANGE,
            "dendrite_tint_spread": TINT_SPREAD,
            "dendrite_tilt_deg": 180.0,
            "dendrite_azimuth_deg": 0.0,
            "dendrite_halo_mm": self.halo_mm,
            "dendrite_halo_share": 1.0,
            "dendrite_halo_clearing": self.clearing,
            "dendrite_reach_share": self.reach_share,
            "dendrite_reach_floor": self.reach_floor,
            "dendrite_reach_contrast": self.reach_contrast,
        }

    def summary(self) -> str:
        return f"{self.colour} moss {self.look}, sheath {self.halo_mm:.1f} mm"


def moss_plan(rng: np.random.Generator, outlines: list[str], taken: set[int]) -> set[int]:
    """Which stones of a sheet carry moss: never a pebble or a stone with plumes."""
    ok = [i for i, o in enumerate(outlines) if o not in NO_MOSS and i not in taken]
    return set(rng.permutation(ok)[: round(MOSS_SHARE * len(outlines))].tolist())


def draw_moss(
    rng: np.random.Generator,
    solid: bool,
    centre_width: float,
    floor: float,
    thunder_egg: bool = False,
) -> Moss:
    """One stone's moss, from its centre (solid or not, and its width), floor and
    outline."""
    u = rng.random(8)
    names = list(COLOUR_SHARES)
    total, edge, colour = sum(COLOUR_SHARES.values()), 0.0, names[-1]
    for name in names:
        edge += COLOUR_SHARES[name] / total
        if float(u[0]) < edge:
            colour = name
            break
    if solid:
        look = "G" if u[1] < G_SHARE else "E"
        spacing, amount, density, sticking, upward, start = LOOKS[look]
        halo = SHEATH_SOLID_MM[0] + SHEATH_SOLID_MM[1] * float(u[2])
        reach_share, centre_share = SOLID_REACH_SHARE, SOLID_CENTRE_SHARE
    else:
        c0, cw = C_TO_E
        k = 0.0 if u[1] < PLAIN_C_SHARE else float(np.clip((c0 - centre_width) / cw, 0, 1))
        c, e = LOOKS["C"], LOOKS["E"]
        _, amount, density, sticking, upward, _ = [
            (1 - k) * a + k * b for a, b in zip(c, e, strict=True)
        ]
        look = f"C>{k:.1f}"
        spacing = C_SPACING_MM[0] * C_SPACING_MM[1] ** float(u[3])
        start = C_START[0] + C_START[1] * float(u[4])
        h0, h1, hk = SHEATH_SHORT_MM
        halo = h0 + h1 * float(u[2]) + hk * k
        # At least a little: a share of 0 would switch the local reach off (ADR 0086).
        reach_share = max(SHORT_REACH_SHARE * (1 - centre_width), 0.05)
        centre_share = SHORT_CENTRE_SHARE
    w0, w1, lean = WRAP_RATE
    return Moss(
        look=look,
        colour=cast(MossColour, colour),
        tint=TINTS[colour],
        spacing_mm=float(spacing),
        amount=float(amount),
        density=float(density),
        sticking=float(sticking),
        upward=float(upward),
        start_height=float(start),
        halo_mm=float(halo),
        clearing=CLEARING[0] + CLEARING[1] * float(u[5]),
        wrap_rate=w0 + w1 * float(u[6]) ** lean,
        reach_share=float(reach_share),
        reach_floor=float(floor),
        centre_share=centre_share,
        reach_contrast=THUNDER_EGG_CONTRAST if thunder_egg else REACH_CONTRAST,
    )
