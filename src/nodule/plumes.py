"""Plumes: agatesim's branched inclusions, grown before the bands (ADR 0018 s.13).

A plume stone runs agatesim's dendrite stage for a moment at the start of growth, so
the bands that follow wrap each plume (the owner's A3; [@pabian1994, pp. 13, 23]
*plausible*). Each plume is traced, so its tint can change from base to tips, differ
a little from plume to plume, and carry a clear halo (agatesim ADR 0079).

The owner's choices (2026-11-09, ADR 0018 s.13); all values `effective`:
- about 1 stone in 4 on mock-up sheets while the look is honed (later 1 in 8-10);
  never the small unbanded pebble;
- look L (dense tufts) by default, R (sparse, fine) on a third of plume stones (the
  third is an assumption, accepted by the owner);
- direction: 3 in 4 up, 1 in 4 sideways;
- colour: Kile's mix [@kile2002, pp. 259-260] (*plausible*), with green and mixed
  ("autumn") stones added on about 1 plume stone in 10 after [@salimi2019]
  (*plausible*; the share is an assumption);
- halos on about 3 in 4 plume stones, 0.3-1 mm wide (an assumption: no source gives a
  width), on a drawn share of their plumes;
- bands grow round plumes slowly, at 0.15-0.3 of their rate from the wall, drawn per
  stone leaning low (`WRAP_RATE`), so they wrap them in thin coats (B, 2026-11-09).

Colour: agatesim stores a plume's tint as one number (0 green Fe-silicate, 0.5
red-brown Fe oxide, 1 black Mn oxide). Nodule reads it through its own ramp
(`INCLUSION_RAMP`), which adds orange-gold between green and red; moss will use the
same ramp. The ramp's colours are by eye (`effective`).

Appearance only: never evidence for a formation mechanism (ADR 0002).
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from agatesim.models.growth import TIME_STEPS
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

Look = Literal["dense", "sparse"]
PlumeColour = Literal["black-brown", "orange-black", "orange-gold", "red", "green", "autumn"]

PLUME_SHARE = 1 / 4  # of a sheet's stones (the owner, while the look is honed)
NO_PLUMES = ("pebble",)  # small and nearly unbanded (ADR 0018 s.7)
SPARSE_SHARE = 1 / 3  # look R among plume stones (assumption)
SIDEWAYS_SHARE = 1 / 4
HALO_STONES = 3 / 4
HALO_MM = (0.3, 1.0)  # assumption (ADR 0018 s.13)
HALO_PLUMES = (0.5, 1.0)  # share of a halo stone's plumes with a halo (assumption)

# Kile's mix [@kile2002, pp. 259-260]: about half black-brown, a third orange base to
# black tips, the rest orange-gold with red about 1 in 20; scaled to 0.9 and joined by
# green and autumn stones, 1 in 20 each (the owner, 2026-11-09).
COLOUR_SHARES: dict[str, float] = {
    "black-brown": 0.45,
    "orange-black": 0.30,
    "orange-gold": 0.105,
    "red": 0.045,
    "green": 0.05,
    "autumn": 0.05,
}
# Per colour: base tint range and change from base to tips (agatesim's tint scale).
TINTS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "black-brown": ((0.82, 0.95), (0.0, 0.08)),
    "orange-black": ((0.28, 0.35), (0.55, 0.65)),
    "orange-gold": ((0.26, 0.34), (0.0, 0.05)),
    "red": ((0.46, 0.52), (0.0, 0.0)),
    "green": ((0.0, 0.08), (0.0, 0.05)),
    "autumn": ((0.0, 0.05), (0.4, 0.5)),  # green base through orange-gold to red tips
}
TINT_SPREAD = 0.05  # plumes in one stone differ a little (the owner)
AUTUMN_SPREAD = 0.15  # several colours in one stone [@salimi2019]

# The probe's settings (ADR 0018 s.13; 8 M cells: L 33 s, R 17 s, peak 0.61 GB).
LOOKS: dict[str, tuple[float, float]] = {"dense": (0.15, 0.5), "sparse": (0.02, 0.15)}
SPACING_MM = 14.0
AMOUNT = 0.02
UPWARD = 0.9
START_HEIGHT = 0.45
CELLS_M = 8.0
# Bands grow round plumes at this share of their growth from the wall (agatesim ADR
# 0080): thin coats, so plumes reaching into the cavity keep its centre and do not each
# carry a set of rings. An assumption chosen by eye; no source gives a rate (ADR 0018
# s.13: at 1, no quartz-lined plume stone kept its centre; at 0.15 one kept it whole, at
# 0.3 a third of it). Drawn per stone over WRAP_RATE, leaning to the low end (the owner,
# 2026-11-09: wider diversity of emergent forms): 0.15 + 0.15 u^WRAP_LEAN, median 0.20.
WRAP_RATE = (0.15, 0.3)
WRAP_LEAN = 1.5
# Reach (agatesim ADR 0081): in a stone with a quartz or open centre, a plume whose tip
# lies farther from the wall than (1 - wrap) x the chalcedony's thickness gets a coat that
# bulges past where the centre begins, and quartz grows on the bulge (seen on the owner's
# look sheet, 2026-11-09: quartz islands in the cavity). So plumes stop short of it, by
# a margin for the bands' wobble and rounding. `effective`; an assumption.
REACH_MARGIN_MM = 1.0
# And in every stone, solid or not, plumes stop at this share of the way from the wall
# to the centre (the owner, 2026-11-09, on a plume tip ringed by its coat beside a
# stone's centre: "stop plumes short of the center"; may be revisited). `effective`.
CENTRE_SHARE = 2 / 3
# The dendrite stage's span, a share of the run: two time steps. No band grows
# during it; `Growth.with_plumes` lengthens the run to match, so the walls are
# unchanged.
STAGE = 2 / TIME_STEPS

# Nodule's ramp over agatesim's tint (stand-in sRGB, by eye). Orange-gold and
# yellow-brown at 0.3 (limonite/goethite), red at 0.5 (hematite), black at 1 (Mn).
INCLUSION_RAMP: tuple[tuple[float, tuple[float, float, float]], ...] = (
    (0.0, (72.0, 104.0, 60.0)),
    (0.3, (196.0, 132.0, 44.0)),
    (0.5, (148.0, 48.0, 28.0)),
    (0.75, (72.0, 42.0, 28.0)),
    (1.0, (28.0, 24.0, 22.0)),
)


class Plumes(BaseModel):
    """One stone's plumes (agatesim dendrite settings, ADR 0079)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    look: Look
    colour: PlumeColour
    tint: float = Field(ge=0, le=1, description="Base tint, agatesim's scale.")
    tint_change: float = Field(ge=-1, le=1, description="Base to tips.")
    tint_spread: float = Field(ge=0, le=0.5, description="Plume to plume.")
    tilt_deg: float = Field(ge=0, le=180, description="0 up, 90 sideways, 180 down.")
    azimuth_deg: float = Field(ge=0, lt=360)
    halo_mm: float = Field(ge=0)
    halo_share: float = Field(ge=0, le=1)
    wrap_rate: float = Field(gt=0, le=1, description="Band growth round plumes / wall's.")

    def model_params(self) -> dict[str, object]:
        density, sticking = LOOKS[self.look]
        params: dict[str, object] = {
            "dendrite_spacing_mm": SPACING_MM,
            "dendrite_amount": AMOUNT,
            "dendrite_density": density,
            "dendrite_sticking": sticking,
            "dendrite_upward": UPWARD,
            "dendrite_start_height": START_HEIGHT,
            "dendrite_cells_3d": CELLS_M,
            "dendrite_wrap_rate": self.wrap_rate,
            "dendrite_tint": self.tint,
            "dendrite_tint_change": self.tint_change,
            "dendrite_tint_spread": self.tint_spread,
            "dendrite_tilt_deg": self.tilt_deg,
            "dendrite_azimuth_deg": self.azimuth_deg,
        }
        if self.halo_mm > 0:
            params |= {"dendrite_halo_mm": self.halo_mm, "dendrite_halo_share": self.halo_share}
        return params

    def summary(self) -> str:
        way = "" if self.tilt_deg == 0 else " sideways"
        halo = f", halo {self.halo_mm:.1f} mm" if self.halo_mm > 0 else ""
        look = "" if self.look == "dense" else " sparse"
        return f"{self.colour}{look} plumes{way}{halo}, wrap {self.wrap_rate:.2f}"


def plume_plan(rng: np.random.Generator, outlines: list[str]) -> set[int]:
    """Which stones of a sheet carry plumes."""
    ok = [i for i, o in enumerate(outlines) if o not in NO_PLUMES]
    return set(rng.permutation(ok)[: round(PLUME_SHARE * len(outlines))].tolist())


def draw_plumes(rng: np.random.Generator) -> Plumes:
    names = list(COLOUR_SHARES)
    p = np.array([COLOUR_SHARES[k] for k in names])
    colour = names[int(rng.choice(len(names), p=p / p.sum()))]
    (b0, b1), (c0, c1) = TINTS[colour]
    tint = float(rng.uniform(b0, b1))
    change = float(rng.uniform(c0, c1))
    look: Look = "sparse" if rng.random() < SPARSE_SHARE else "dense"
    sideways = rng.random() < SIDEWAYS_SHARE
    azimuth = float(90 * rng.integers(0, 4)) if sideways else 0.0
    halo = rng.random() < HALO_STONES
    halo_mm = float(rng.uniform(*HALO_MM)) if halo else 0.0
    share = float(rng.uniform(*HALO_PLUMES)) if halo else 1.0
    lo, hi = WRAP_RATE
    wrap = lo + (hi - lo) * float(rng.random()) ** WRAP_LEAN
    return Plumes(
        look=look,
        colour=cast(PlumeColour, colour),
        tint=tint,
        tint_change=change,
        tint_spread=AUTUMN_SPREAD if colour == "autumn" else TINT_SPREAD,
        tilt_deg=90.0 if sideways else 0.0,
        azimuth_deg=azimuth,
        halo_mm=halo_mm,
        halo_share=share,
        wrap_rate=wrap,
    )


def inclusion_rgb(tint: NDArray[np.floating]) -> NDArray[np.float64]:
    """Stand-in colour (n, 3) of inclusion points from their tint."""
    t = np.clip(np.nan_to_num(np.asarray(tint, dtype=np.float64), nan=1.0), 0.0, 1.0)
    xs = [x for x, _ in INCLUSION_RAMP]
    return np.stack(
        [np.interp(t, xs, [c[k] for _, c in INCLUSION_RAMP]) for k in range(3)], axis=-1
    )


def reach_mm(centre_mm: float, wrap_rate: float, wobble_mm: float, solid: bool) -> float:
    """How far from the wall plumes may grow, in a stone whose centre lies `centre_mm`
    of wall growth in: CENTRE_SHARE of that; and where the centre is quartz or open,
    no farther than lets their coats (wrap_rate x that growth) stay clear of it, by
    REACH_MARGIN_MM plus the bands' wobble. At least half a millimetre."""
    reach = CENTRE_SHARE * centre_mm
    if not solid:
        reach = min(reach, (1 - wrap_rate) * centre_mm - REACH_MARGIN_MM - wobble_mm)
    return max(reach, 0.5)
