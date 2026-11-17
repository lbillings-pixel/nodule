"""Host rind and host specks on preview pictures (work plan item 13; ADR 0018 s.16).

The rind is a shell outside the stone's wall, drawn at a distance d = -level / |grad
level| (the solid's level divided by its slope: probe a, `runs/host-probe/`), with its
width and textures from hashed value noise and hashed grains in WORLD coordinates, so
no lattice is stored and a preview matches a full-resolution cut (CLAUDE.md). A rind
thinner than a pixel is drawn by the share of each pixel's span that it covers.

The owner's choices (look sheet, 2026-11-17; ADR 0018 s.16):
- Volcanic stones (vesicle, prolate): always a rind, from the thin green clay rim
  (character 0, about 0.6 mm; [@proust2007], [@morteani2010]: *verified*) to the
  mottled brown-green shell with host-coloured bites at its outer edge (character 1,
  about 1.5 mm; the owner's eye on basalt, an assumption; 4 mm until the owner's
  render verdict of 2026-11-17), about two thirds of the stones at the thin end (the
  owner, 2026-11-17). Width capped at 3 mm by the owner (6 mm before, after
  [@morteani2010]); it varies along the wall by at most 2x the stone's typical width. On
  about 1 nodule in 13 (the owner's experience) a window: a patch of the wall with no
  rind, at a random place (the owner kept it random, 2026-11-17).
- Rind colour: half the stones echo their own colour class, the rest draw one of five
  muted colours (the owner's observation; *plausible* as a look).
- Sediment stones (irregular, pebble): on about half, option A, a pale tan host zone
  (flint cortex [@degrooth2017], *verified* for flint; the tan is the owner's
  observation), 0.05-1 mm, typically about 0.2 mm.
- Thunder eggs: no rind (it belongs on the ball's outside, not modelled); rhyolite
  specks in the host [@goetze2024] (*verified* 5-20 %, about 1 mm; the owner asked for
  them smaller and fewer: 0.5-1.0 mm, about 5 % of the host face).
- Basalt: sparse filled holes with green rims (*plausible* [@morteani2010]; white
  cores an assumption); sediment host lightly mottled (an assumption).

Every number here is `effective` unless marked otherwise. No bleed into the host (the
owner: skip it). The renderer draws no rind or specks: a preview shows them, a render
does not yet. Appearance only: never evidence for a formation mechanism (ADR 0002).
"""

from __future__ import annotations

import math
from typing import Literal, Protocol, cast

import numpy as np
from agatesim.grid import FloatArray
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

RindKind = Literal["volcanic", "pale"]
Tone = Literal["echo", "green", "brown-green", "brown", "red-brown", "yellow-ochre", "tan"]

VOLCANIC = ("vesicle", "prolate")
SEDIMENT = ("irregular", "pebble")

# Recipe shares and sizes (ADR 0018 s.16, the owner's verdict 2026-11-17).
PALE_CHANCE = 0.5  # sediment stones with the pale rind (owner: "sometimes"; proposed half)
PALE_MM = (0.05, 1.0)  # log-uniform: median about 0.22 mm (owner's observation)
PALE_SPREAD = 1.6  # the pale rind's widest, as a multiple of its typical width
# Rind width at character 0 and 1. The shell end was 4 mm on the look sheet; the owner
# (2026-11-17, sheet 1 stone 6's render) judged thick, uneven rinds rare edge cases that
# look unrealistic as drawn: typical 1.5 mm or less, 3 mm at most.
THIN_MM, SHELL_MM = 0.6, 1.5
# Character u**THIN_BIAS for u uniform: two thirds of the stones below 0.25, the thin-rim
# end (the owner, 2026-11-17: "about two-thirds"; 0.25 ** (1 / 3.42) = 0.667).
THIN_BIAS = 3.42
ECHO_CHANCE = 0.5
WINDOW_CHANCE = 1 / 13.5  # owner: about 1 nodule in 12-15
WINDOW_MM = (1.5, 4.0)  # window radius on the wall, log-uniform (assumption)
WINDOW_RAMP_MM = 0.6  # the rind thins to nothing over this distance at a window's edge
# "barely reaches 5 mm ... only in very rare cases 2 cm" [@morteani2010]; the owner's cap
# for the drawn rind, at any point, is 3 mm (2026-11-17; was 6).
VOLCANIC_CAP_MM = 3.0
# Along the wall: lognormal width, sigma WIDTH_SIGMA[0] + WIDTH_SIGMA[1] x character, at
# most WIDEST x the stone's typical width (was 0.15 + 0.35 c and 2.5 x; the owner: less
# variable, 2026-11-17).
WIDTH_SIGMA = (0.15, 0.15)
WIDEST = 2.0

ECHO: dict[str, tuple[int, int, int]] = {  # a muted rind colour by the stone's colour class
    "neutral": (118, 116, 104),
    "iron": (122, 74, 48),
    "onyx": (66, 60, 54),
    "stained": (112, 88, 58),
    "carnelian": (134, 72, 44),
}
DRAWN: dict[str, tuple[int, int, int]] = {  # green from the sources; the rest the owner's
    "green": (68, 88, 58),
    "brown-green": (96, 92, 58),
    "brown": (102, 78, 54),
    "red-brown": (116, 64, 48),
    "yellow-ochre": (148, 122, 66),
}
TAN = (198, 178, 142)  # the sediment rind: tan-beige (owner's observation)

E_MM = 0.01  # central differences for the level's slope
U64 = np.uint64
MASK = (1 << 64) - 1
# Standard deviation of one octave of `value_noise` (measured over 200,000 points, three
# keys: 0.3700-0.3708; `tests/test_rind.py` re-measures it).
VALUE_STD = 0.3703
HOST_KEY = 0x13_0B_1D  # the host drawing's key (mixed with the stone's seed)


class Rind(BaseModel):
    """One stone's host rind."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: RindKind
    thickness_mm: float = Field(
        gt=0, le=VOLCANIC_CAP_MM, description="The stone's typical rind width."
    )
    character: float = Field(
        default=0.0, ge=0, le=1, description="Volcanic: 0 thin clay rim .. 1 mottled shell."
    )
    tone: Tone
    window: tuple[float, float, float] | None = Field(
        default=None, description="Unit direction from the solid's centre to a window."
    )
    window_mm: float = Field(default=2.5, gt=0, le=10)

    @model_validator(mode="after")
    def _consistent(self) -> Rind:
        if (self.kind == "pale") != (self.tone == "tan"):
            raise ValueError("the pale rind is tan, and only the pale rind")
        if self.window is not None:
            if self.kind != "volcanic":
                raise ValueError("windows only on volcanic rinds")
            if abs(math.hypot(*self.window) - 1.0) > 1e-6:
                raise ValueError("a window's direction must be a unit vector")
        return self

    def colour(self, colour_class: str) -> tuple[int, int, int]:
        if self.tone == "echo":
            return ECHO[colour_class]
        return TAN if self.tone == "tan" else DRAWN[self.tone]

    def reach_mm(self) -> float:
        """The widest the rind can be drawn (mm)."""
        if self.kind == "volcanic":
            return min(WIDEST * self.thickness_mm, VOLCANIC_CAP_MM) + 0.2
        return PALE_SPREAD * self.thickness_mm + 0.05

    def summary(self) -> str:
        if self.kind == "pale":
            return f"tan rind {self.thickness_mm:.2f} mm"
        shell = " shell" if self.character >= 0.5 else ""
        win = " + window" if self.window is not None else ""
        return f"{self.tone} rind{shell} {self.thickness_mm:.1f} mm{win}"


def volcanic_width(character: float) -> float:
    """The typical width at a character: log-linear from THIN_MM to SHELL_MM."""
    return math.exp(math.log(THIN_MM) + character * (math.log(SHELL_MM) - math.log(THIN_MM)))


def draw_rind(rng: np.random.Generator, outline: str) -> Rind | None:
    """One stone's rind, or none. Every stone draws the same numbers whatever its
    outline, so a later change of eligibility moves no other stone's draw."""
    u = rng.random(8)
    n = rng.standard_normal(3)
    if outline in VOLCANIC:
        c = float(u[0]) ** THIN_BIAS
        names = list(DRAWN)
        tone = "echo" if u[1] < ECHO_CHANCE else names[min(int(u[2] * len(names)), len(names) - 1)]
        window = None
        if u[3] < WINDOW_CHANCE:
            d = n / float(np.linalg.norm(n))
            window = (float(d[0]), float(d[1]), float(d[2]))
        lo, hi = WINDOW_MM
        return Rind(
            kind="volcanic",
            thickness_mm=volcanic_width(c),
            character=c,
            tone=cast(Tone, tone),
            window=window,
            window_mm=math.exp(math.log(lo) + float(u[4]) * (math.log(hi) - math.log(lo))),
        )
    if outline in SEDIMENT and u[5] < PALE_CHANCE:
        lo, hi = PALE_MM
        width = math.exp(math.log(lo) + float(u[6]) * (math.log(hi) - math.log(lo)))
        return Rind(kind="pale", thickness_mm=width, tone="tan")
    return None


# ---------------------------------------------------------------- hashed noise


def _mix(x: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """splitmix64 finaliser on uint64 arrays (wraps silently)."""
    with np.errstate(over="ignore"):
        x = (x ^ (x >> U64(30))) * U64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> U64(27))) * U64(0x94D049BB133111EB)
    return x ^ (x >> U64(31))


def hash3(i: NDArray[np.int64], key: int) -> NDArray[np.uint64]:
    """One uint64 per integer lattice cell (rows of i: n x 3)."""
    k = U64((key * 0x9E3779B97F4A7C15) & MASK)
    h = _mix(i[:, 0].astype(np.uint64) ^ k)
    h = _mix(h ^ i[:, 1].astype(np.uint64))
    return _mix(h ^ i[:, 2].astype(np.uint64))


def unit(h: NDArray[np.uint64], salt: int = 0) -> FloatArray:
    """Uniform [0, 1) from a hash (salted for independent draws per cell)."""
    if salt:
        h = _mix(h ^ U64((salt * 0xD1B54A32D192ED03) & MASK))
    return (h >> U64(11)).astype(np.float64) / float(1 << 53)


def value_noise(points: FloatArray, scale: float, key: int) -> FloatArray:
    """Smooth value noise in [-1, 1] at world points, cells of side `scale` mm."""
    p = points / scale
    i0 = np.floor(p).astype(np.int64)
    f = p - i0
    u = f * f * (3 - 2 * f)
    out = np.zeros(len(p))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                v = 2 * unit(hash3(i0 + np.array([dx, dy, dz]), key)) - 1
                w = (u[:, 0] if dx else 1 - u[:, 0]) * (u[:, 1] if dy else 1 - u[:, 1])
                out += v * w * (u[:, 2] if dz else 1 - u[:, 2])
    return out


def noise(points: FloatArray, scale: float, key: int) -> FloatArray:
    """About unit-variance smooth noise in world mm, two octaves."""
    a = value_noise(points, scale, key)
    b = value_noise(points, scale / 2.1, key + 1)
    return (a + 0.5 * b) / (VALUE_STD * math.sqrt(1.25))


def _frame(h: NDArray[np.uint64]) -> FloatArray:
    """A hashed random orthonormal frame per grain (n, 3, 3), rows the axes."""
    a = np.stack([2 * unit(h, s) - 1 for s in (20, 21, 22)], 1) + 1e-9
    b = np.stack([2 * unit(h, s) - 1 for s in (23, 24, 25)], 1)
    e1 = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b - np.sum(b * e1, 1, keepdims=True) * e1 + 1e-9
    e2 = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.stack([e1, e2, np.cross(e1, e2)], 1)


Shape = tuple[float, tuple[float, float, float], float]  # share, half-axes, p-norm


def grains(
    points: FloatArray,
    cell: float,
    key: int,
    p: float,
    r0: float,
    r1: float,
    shapes: list[Shape] | None = None,
) -> tuple[NDArray[np.bool_], FloatArray, NDArray[np.uint64]]:
    """Hashed grains in world mm: at most one per cell of side `cell` (present with
    chance p), centre jittered, size U(r0, r1). Without `shapes`, balls; with them,
    each grain draws a kind (share, half-axes as multiples of its size, p-norm: 2
    round, 6 blocky) and a random orientation. Returns (hit, rel, id): rel < 1 inside
    (1 at the grain's edge), id the grain's hash."""
    reach = r1 * (max(max(a) * 3 ** (1 / pn) for _, a, pn in shapes) if shapes else 1.0)
    if reach >= cell:
        raise ValueError("a grain must not reach two cells away")
    q = points / cell
    base = np.floor(q).astype(np.int64)
    hit = np.zeros(len(q), bool)
    rel = np.full(len(q), np.inf)
    ids = np.zeros(len(q), np.uint64)
    table = shapes or [(1.0, (1.0, 1.0, 1.0), 2.0)]
    cum = np.cumsum(np.array([s for s, _, _ in table], dtype=np.float64))
    axes = np.array([a for _, a, _ in table], dtype=np.float64)
    pns = np.array([pn for _, _, pn in table], dtype=np.float64)
    for off in np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]), -1).reshape(-1, 3):
        c = base + off
        h = hash3(c, key)
        present = unit(h, 1) < p
        centre = (c + 0.15 + 0.7 * np.stack([unit(h, s) for s in (2, 3, 4)], 1)) * cell
        r = r0 + (r1 - r0) * unit(h, 5)
        d: FloatArray
        if shapes:
            pick = unit(h, 9) * float(cum[-1])
            kind = np.minimum(np.searchsorted(cum, pick, side="right"), len(cum) - 1)
            loc: FloatArray = np.abs(np.einsum("nij,nj->ni", _frame(h), points - centre))
            loc = loc / (r[:, None] * axes[kind])
            pn: FloatArray = pns[kind]
            d = np.power(np.sum(np.power(loc, pn[:, None]), 1), 1 / pn)
        else:
            d = np.asarray(np.linalg.norm(points - centre, axis=1) / r, dtype=np.float64)
        better = present & (d < 1) & (d < rel)
        hit |= better
        rel = np.where(better, d, rel)
        ids = np.where(better, h, ids)
    return hit, rel, ids


# ---------------------------------------------------------------- distance


class Levelled(Protocol):
    def level(self, points: FloatArray) -> FloatArray: ...  # agatesim's Solid has it


def wall_distance(
    solid: Levelled, points: FloatArray, band: float
) -> tuple[FloatArray, FloatArray]:
    """Signed distance to the wall (+ outside), -level / |grad level|, where it lies
    within `band` mm (else +-inf), and the unit inward gradient there (towards the
    wall from outside). The slope varies by place (0.80-1.40 on sheets 0-1), so the
    level alone would make a rind up to 25 % thicker at an egg's long ends (s.16)."""
    lev = solid.level(points)
    d = np.where(lev > 0, -np.inf, np.inf)
    g_in = np.zeros_like(points)
    # |grad| < 1.5 on every stone measured, so |level| < 1.5 band covers the band.
    near = np.flatnonzero(np.abs(lev) < 1.5 * band)
    pn = points[near]
    g = np.stack(
        [(solid.level(pn + E_MM * e) - solid.level(pn - E_MM * e)) / (2 * E_MM) for e in np.eye(3)],
        1,
    )
    gn = np.maximum(np.linalg.norm(g, axis=1), 1e-9)
    dn = -lev[near] / gn
    keep = np.abs(dn) < band
    d[near[keep]] = dn[keep]
    g_in[near[keep]] = (g / gn[:, None])[keep]
    return d, g_in


# ---------------------------------------------------------------- drawing

# Rhyolite phenocrysts: blocky cream feldspar, rounded glassy-grey quartz, a few dark
# flakes; 0.5-1.0 mm across (owner: x0.87 size, chance 0.7). They cover 4.8 % of the
# host on average (1,000,000 random points, three keys; one face varies), the low edge
# of the 5-20 % of [@goetze2024]; the probe's 15 % and 7 % came from a small patch.
RHYOLITE_SHAPES: list[Shape] = [
    (0.62, (1.3, 0.8, 0.55), 6.0),
    (0.32, (1.0, 1.0, 1.0), 2.5),
    (0.06, (0.7, 0.7, 0.15), 6.0),
]
RHYOLITE_GRAINS = (1.6, 0.7, 0.3 * 0.87, 0.6 * 0.87)  # cell mm, chance, size range mm
RHYOLITE_RGB = np.array([(196, 182, 158), (136, 130, 124), (52, 44, 38)], float)
# Basalt: sparse filled holes, radius 0.5-1.2 mm (assumption; the geode level is
# vesicle-free [@proust2007]); green rims (*plausible*), cores grey agate, white
# (assumed) or rust.
HOLES = (6.0, 0.2, 0.5, 1.2)
HOLE_CORE_RGB = np.array([(172, 172, 166), (222, 222, 216), (132, 84, 44)], float)


def host_specks(rgb: FloatArray, points: FloatArray, host: str, seed: int) -> None:
    """Specks on host pixels in place: rgb (n, 3) float, points (n, 3) world mm, all
    outside the stone. host: "thunder-egg", "volcanic" or "sedimentary"."""
    if not len(points):
        return
    k = HOST_KEY * 7919 + seed
    if host == "thunder-egg":
        cell, p, r0, r1 = RHYOLITE_GRAINS
        hit, _, ids = grains(points, cell, k + 1, p, r0, r1, RHYOLITE_SHAPES)
        u = unit(ids, 9)
        kinds = np.where(u < 0.62, 0, np.where(u < 0.94, 1, 2))
        rgb[hit] = RHYOLITE_RGB[kinds[hit]]
    elif host == "volcanic":
        cell, p, r0, r1 = HOLES
        hit, rel, ids = grains(points, cell, k + 2, p, r0, r1)
        u = unit(ids, 9)
        kind = np.where(u < 0.4, 0, np.where(u < 0.7, 1, 2))
        rim_w = 0.18 + 0.15 * unit(ids, 10)
        rim_rgb = np.array([60, 82, 52]) + unit(ids, 11)[:, None] * np.array([70, 70, 55])
        rim = hit & (rel > 1 - rim_w)
        rgb[hit] = HOLE_CORE_RGB[kind[hit]]
        rgb[rim] = rim_rgb[rim]
    else:  # light mottle (assumption)
        m = 0.05 * noise(points, 3.0, k + 3) + 0.025 * noise(points, 0.5, k + 4)
        rgb *= 1 + m[:, None]


def rind_width(rind: Rind, foot: FloatArray, seed: int, window_at: FloatArray | None) -> FloatArray:
    """The rind's width (mm) at wall points `foot`: lognormal along the wall, rougher
    on the shell end, nothing at a window."""
    k = HOST_KEY * 104729 + seed * 31
    if rind.kind == "volcanic":
        c = rind.character
        sig = WIDTH_SIGMA[0] + WIDTH_SIGMA[1] * c
        w = rind.thickness_mm * np.exp(sig * noise(foot, 4.0, k + 1) - sig**2 / 2)
        w *= 1 + 0.12 * c * noise(foot, 0.8, k + 2)  # rough outer edge on the shell end
        w = np.clip(w, 0, min(WIDEST * rind.thickness_mm, VOLCANIC_CAP_MM))
    else:
        w = rind.thickness_mm * np.exp(0.3 * noise(foot, 3.0, k + 1) - 0.045)
        # inside its reach (1-3 % of the wall would pass it), so the drawn edge is the
        # noise's and not the cut-off band's, which moves with the pixel size
        w = np.clip(w, 0, PALE_SPREAD * rind.thickness_mm)
    if window_at is not None:
        t = np.linalg.norm(foot - window_at, axis=1)
        w = w * np.clip((t - (rind.window_mm - WINDOW_RAMP_MM)) / WINDOW_RAMP_MM, 0, 1)
    return w


def coverage(d: FloatArray, w: FloatArray, px: float) -> FloatArray:
    """Share of each pixel's span [d - px/2, d + px/2] that lies in the rind [0, w]."""
    lo, hi = d - px / 2, d + px / 2
    return np.clip(np.minimum(hi, w) - np.maximum(lo, 0), 0, None) / px


def draw_rind_rgb(
    rgb: FloatArray,
    points: FloatArray,
    inside: NDArray[np.bool_],
    px: float,
    solid: Levelled,
    rind: Rind,
    colour: tuple[int, int, int],
    host_rgb: FloatArray,
    seed: int,
    window_at: FloatArray | None = None,
) -> tuple[NDArray[np.intp], FloatArray, FloatArray]:
    """Overlay the rind on rgb (n, 3) float in place, by coverage. Pixels centred
    outside the stone draw their agate share as host, as without a rind (handing it to
    the rind would draw a thin rind heavy at whole face). Returns the touched indices,
    the width there and each one's rind share."""
    d_all, g_all = wall_distance(solid, points, rind.reach_mm() + px)
    sel = np.flatnonzero(np.isfinite(d_all) & (d_all > -px) & (d_all < rind.reach_mm() + px))
    x, dd = points[sel], d_all[sel]
    foot = x + dd[:, None] * g_all[sel]  # back to the wall along the normal
    w = rind_width(rind, foot, seed, window_at)
    cov = coverage(dd, w, px)
    k = HOST_KEY * 104729 + seed * 31
    base = np.array(colour, float)
    col = np.broadcast_to(base, (len(x), 3)).copy()
    if rind.kind == "volcanic":
        c = rind.character
        # mottle into a second tone: brown on a green rind, green on the others, so
        # the shell end reads brown-green
        other = DRAWN["brown"] if base[1] >= base[0] else DRAWN["green"]
        alt = 0.85 * np.array(other, float)
        m = np.clip(0.5 + 0.6 * noise(x, 1.5, k + 3), 0, 1) * c
        col = col * (1 - m[:, None]) + alt * m[:, None]
        col *= 1 + (0.04 + 0.06 * c) * noise(x, 0.25, k + 4)[:, None]  # clay grain
        # host-coloured bites in the outer part of the shell (owner's eye)
        outer = np.clip(dd / np.maximum(w, 1e-6), 0, 1)
        bite = (noise(x, 1.0, k + 5) > 2.0 - 1.4 * c * outer) & (outer > 0.45) & (c > 0.2)
        col[bite] = host_rgb[sel][bite]
        # darker at the wall on the thin end (a celadonite rim reads dark green)
        col *= (1 - 0.15 * (1 - c) * np.exp(-np.clip(dd, 0, None) / 0.3))[:, None]
    else:  # porous, gritty
        col *= 1 + 0.07 * noise(x, 0.04, k + 3)[:, None]
        col *= 1 + 0.03 * noise(x, 0.4, k + 4)[:, None]
    ins = inside[sel]
    hostw = np.where(ins, 0.0, 1 - cov)
    keep = np.where(ins, 1 - cov, 0.0)
    out = cov[:, None] * col + hostw[:, None] * host_rgb[sel] + keep[:, None] * rgb[sel]
    touched = (cov > 0) | ~ins
    rgb[sel[touched]] = out[touched]
    return sel, w, cov
