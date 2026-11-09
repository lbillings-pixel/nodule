"""Band-following dust: fine iron oxide along band boundaries (ADR 0018 s.11).

Four forms, varied per stone; the first is the band iron agatesim already gives:

1. *tinted band*: pigment even through a band (nothing added here);
2. *dust line*: a thin strip at a band boundary, darker and more saturated, drawn
   by area coverage so it can be thinner than a pixel; its strength varies along
   the band with smooth noise in world coordinates;
3. *speck string*: specks drawn out along the band, in loose clustered rows on the
   same boundaries;
4. *coarse blob*: rounded aggregates at random in the inner chalcedony.

Sources (graded in ADR 0018 s.11): fine dust along the crystallisation front and
coarse aggregates at random in the inner chalcedony [@goetze2009, pp. 680-81;
@goetze2020, p. 27]; dust mostly on band boundaries [@howard2018, Fig. 5], wall-
parallel in a small part of each band [@mrozik2023, p. 13, Fig. 14]; white bands and
the quartz centre nearly clean [@howard2018; @mrozik2023, p. 23, Table 5;
@goetze2009]. Which side of a boundary the dust takes is drawn per stone (the
owner's option A3). No dust in floor bands: no source describes it. Every share,
size and strength here is `effective`: the sources give no frequencies, so these
are starting values for the owner's eye.

Band boundaries are exact: agatesim's bands are periods of growth time, each a
run of growth steps, so a table of each step's band, probed once per stone, gives
every boundary as a step number; a point's distance to it is its step coordinate
(smooth growth time x steps) less the boundary's, times the step's thickness in mm.

Random choices come from spawned streams of the stone's seed. Specks and blobs sit
in cells fixed to the world: each cell's draws are a counter-based hash of the
stream's seed and the cell, so any point, in any batch or at any resolution, sees
the same specks. The dust lines' pixel values depend on the pixel (area coverage,
an anti-aliasing filter); their dust per unit area does not.

Appearance only: never evidence for a formation mechanism (ADR 0002).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from agatesim.fields import Phase, Pigment
from agatesim.grid import FloatArray
from agatesim.models.growth import TIME_STEPS
from agatesim.volume.models import SmoothNoise3
from agatesim.volume.simulate import VolumeSimulation
from agatesim.volume.solids import BoolArray
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

DustPigment = Literal["hematite", "goethite", "iron"]
Side = Literal["end", "start"]

# Shares of stones per sheet (ADR 0018 s.11, starting values, `effective`).
LINE_SHARE = 1 / 2
SPECK_SHARE = 1 / 4
BLOB_SHARE = 1 / 4
# Dusty share of a stone's coloured band boundaries, near the wall; it falls by
# CENTRE_FADE toward the centre ("fewer dusty bands toward the centre": an assumption
# on its form).
BOUNDARY_SHARE = (0.2, 0.4)
CENTRE_FADE = 0.5
# Dust lines: 0.03-0.06 mm, from figures (*plausible*); strength is the impurity added
# at full cover; patches along the band 5-20 mm (one figure: at least 10 mm).
LINE_MM = (0.03, 0.06)
LINE_STRENGTH = (0.4, 0.8)
PATCH_MM = (5.0, 20.0)
# Specks: 0.03-0.3 mm long [@mrozik2023, Fig. 14b; @howard2018, Fig. 5; sizes measured
# on figures, *speculative*], drawn out along the band. A cell holds at most one
# candidate speck; the cell is as wide as the largest half-length, so a point's own
# cell and its 26 neighbours hold every speck that can reach it.
SPECK_MM = (0.03, 0.3)
SPECK_ELONGATION = (1.5, 3.0)
SPECK_CELL_MM = SPECK_MM[1] / 2
SPECK_FILL = (0.3, 0.8)  # chance a cell holds a candidate, at the heart of a cluster
CLUSTER_MM = (1.0, 3.0)  # noise scale of the clusters along a row
SPECK_STRENGTH = 0.85
# Coarse blobs: 0.1-0.5 mm across [@goetze2009, "several 100 µm"], in the inner
# chalcedony (depth beyond INNER of the chalcedony's run), per cm² of face.
BLOB_MM = (0.1, 0.5)
BLOB_CELL_MM = BLOB_MM[1] / 2
BLOB_PER_CM2 = (1.0, 6.0)
BLOB_STRENGTH = 0.9
INNER = 0.4
# A band counts as white (no dust) if its pigment is white, or it is milky and
# nearly free of iron: an assumption, as agatesim's white is milky chalcedony.
WHITE_TRANSLUCENCY = 0.35
WHITE_IRON = 0.15
PIGMENT_AT = 0.1  # dust this strong sets the point's pigment
PROBE_STEP_MM = 0.005  # probe spacing, well under the thinnest growth step (~0.012 mm)

PIGMENTS: dict[DustPigment, Pigment] = {
    "hematite": Pigment.HEMATITE,
    "goethite": Pigment.GOETHITE,
    "iron": Pigment.IRON,
}


class Dust(BaseModel):
    """One stone's dust: which forms, and their sizes. All `effective`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lines: bool = False
    specks: bool = False
    blobs: bool = False
    pigment: DustPigment = "hematite"
    side: Side = Field(
        default="end", description="end: the inner, younger side of a band; start: the outer."
    )
    boundary_share: float = Field(default=0.3, ge=0, le=1)
    line_mm: float = Field(default=0.045, gt=0)
    strength: float = Field(default=0.6, ge=0, le=1)
    patch_mm: float = Field(default=10.0, gt=0)
    speck_fill: float = Field(default=0.5, ge=0, le=1)
    cluster_mm: float = Field(default=2.0, gt=0)
    blob_per_cm2: float = Field(default=3.0, ge=0)

    def summary(self) -> str:
        forms = [k for k in ("lines", "specks", "blobs") if getattr(self, k)]
        return "dust " + "+".join(forms)


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def draw_dust(
    rng: np.random.Generator, red: bool, *, lines: bool, specks: bool, blobs: bool
) -> Dust:
    """A stone's dust: red stones (hematite in the palette) red, others red, yellow
    or brown (ADR 0018 s.11)."""
    if red:
        pigment: DustPigment = "hematite"
    else:
        pigment = str(rng.choice(["hematite", "goethite", "iron"]))  # type: ignore[assignment]
    return Dust(
        lines=lines,
        specks=specks,
        blobs=blobs,
        pigment=pigment,
        side=str(rng.choice(["end", "start"])),  # type: ignore[arg-type]
        boundary_share=float(rng.uniform(*BOUNDARY_SHARE)),
        line_mm=float(rng.uniform(*LINE_MM)),
        strength=float(rng.uniform(*LINE_STRENGTH)),
        patch_mm=_log_uniform(rng, *PATCH_MM),
        speck_fill=float(rng.uniform(*SPECK_FILL)),
        cluster_mm=_log_uniform(rng, *CLUSTER_MM),
        blob_per_cm2=_log_uniform(rng, *BLOB_PER_CM2),
    )


# --- the band table ---------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class BandTable:
    """Each growth step's band (-1 for none), whether that band is white, and the
    step's thickness (mm, at the walls). Steps no probe point reached take their
    predecessor's band."""

    band: NDArray[np.int64]
    white: NDArray[np.bool_]
    step_mm: FloatArray

    @classmethod
    def probe(cls, volume: VolumeSimulation, step_mm: FloatArray) -> BandTable:
        """Read bands along 26 rays from the solid's centre, PROBE_STEP_MM apart."""
        solid = volume.solid
        centre = np.asarray(solid.centre(), dtype=np.float64)
        lo, hi = solid.bounds()
        reach = float(np.max(np.asarray(hi) - np.asarray(lo)))
        dirs = np.array(
            [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)],
            dtype=np.float64,
        )
        dirs = dirs[np.any(dirs != 0, axis=1)]
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        s = np.arange(0.0, reach, PROBE_STEP_MM)
        points = (centre + dirs[:, None, :] * s[None, :, None]).reshape(-1, 3)
        f = volume.evaluate(points, solid.contains(points))
        ok = (f.phase == Phase.CHALCEDONY) & (f.band >= 0) & np.isfinite(f.t_dep)
        k = np.clip(np.floor(f.t_dep[ok] * TIME_STEPS).astype(np.int64), 0, TIME_STEPS - 1)
        trans = f.band_translucency if f.band_translucency is not None else f.translucency
        imp = f.band_impurity if f.band_impurity is not None else f.impurity
        pig = f.pigment[ok] if f.pigment is not None else np.zeros(int(ok.sum()), np.uint8)
        is_white = (pig == Pigment.WHITE) | (
            (trans[ok] < WHITE_TRANSLUCENCY) & (imp[ok] < WHITE_IRON)
        )
        band = np.full(TIME_STEPS, -1, dtype=np.int64)
        white = np.zeros(TIME_STEPS, dtype=bool)
        band[k] = f.band[ok]
        white[k] = is_white  # one band's draws are the same at every point
        # A gap between two reached chalcedony steps takes the earlier band, so its
        # boundary lies at the next reached step (gaps are rare at this spacing).
        reached = np.flatnonzero(band >= 0)
        for a, b in itertools.pairwise(reached.tolist()):
            if b > a + 1:
                band[a + 1 : b] = band[a]
                white[a + 1 : b] = white[a]
        return cls(band, white, np.asarray(step_mm, dtype=np.float64))

    def boundaries(self) -> NDArray[np.int64]:
        """Steps e at which a new band starts, with chalcedony bands on both sides."""
        b = self.band
        e = np.flatnonzero((b[1:] != b[:-1]) & (b[1:] >= 0) & (b[:-1] >= 0)) + 1
        return e.astype(np.int64)


# --- world-fixed cells -------------------------------------------------------------------

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_PRIMES = (
    np.uint64(0x8CB92BA72F3D8DD7),
    np.uint64(0xD6E8FEB86659FD93),
    np.uint64(0xA0761D6478BD642F),
)


def _mix(z: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """splitmix64's finaliser: a bijection that scatters every input bit."""
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


def cell_uniforms(seed: int, cells: NDArray[np.int64], k: int) -> FloatArray:
    """k uniforms in [0, 1) per integer cell (n, 3), a pure function of seed and cell."""
    with np.errstate(over="ignore"):
        h = np.full(len(cells), np.uint64(seed) * _GOLDEN, dtype=np.uint64)
        for axis in range(3):
            h = _mix(h ^ (cells[:, axis].astype(np.int64).view(np.uint64) * _PRIMES[axis]))
        out = np.empty((len(cells), k))
        for j in range(k):
            z = _mix(h + np.uint64(j + 1) * _GOLDEN)
            out[:, j] = (z >> np.uint64(11)).astype(np.float64) * 2.0**-53
    return out


_OFFSETS = np.array(
    [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)], dtype=np.int64
)


def _overlap(
    lo: FloatArray, hi: FloatArray, a: FloatArray | float, b: FloatArray | float
) -> FloatArray:
    return np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None)


def strip_cover(delta: FloatArray, width: FloatArray | float, half_px: float) -> FloatArray:
    """Share of a pixel (half-width half_px, mm, across the band) that a strip
    0 <= delta < width covers; with half_px 0, a point sample."""
    if half_px <= 0:
        return ((delta >= 0) & (delta < width)).astype(np.float64)
    return _overlap(delta - half_px, delta + half_px, 0.0, width) / (2 * half_px)


def disc_cover(r: FloatArray, radius: FloatArray, pixel_mm: float) -> FloatArray:
    """Soft edge over one pixel, so small specks fade rather than flicker."""
    if pixel_mm <= 0:
        return (r < radius).astype(np.float64)
    return np.clip((radius - r) / pixel_mm + 0.5, 0.0, 1.0)


# --- the layer ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class DustLayer:
    """A stone's dust at world points. `amount` gives each form's cover (0..1)."""

    dust: Dust
    table: BandTable
    dusty: NDArray[np.int64]  # boundary steps that carry dust
    start: float  # growth time at which the walls start (no dust before: the floor)
    chalcedony_end: float
    patch_noise: SmoothNoise3
    cluster_noise: SmoothNoise3
    speck_seed: int
    blob_seed: int

    @classmethod
    def of(
        cls,
        dust: Dust,
        volume: VolumeSimulation,
        step_mm: FloatArray,
        start: float,
        chalcedony_end: float,
        seed: int,
    ) -> DustLayer:
        choose, patch, cluster, cells = np.random.SeedSequence(seed).spawn(6)[5].spawn(4)
        table = BandTable.probe(volume, step_mm)
        e = table.boundaries()
        host = e if dust.side == "start" else e - 1
        depth = np.clip((e / TIME_STEPS - start) / max(chalcedony_end - start, 1e-9), 0, 1)
        u = np.random.default_rng(choose).uniform(size=TIME_STEPS)[: len(e)]
        eligible = ~table.white[host] & (e > math.ceil(start * TIME_STEPS))  # not the floor
        keep = eligible & (u < dust.boundary_share * (1 - CENTRE_FADE * depth))
        if (dust.lines or dust.specks) and not keep.any() and eligible.any():
            # A stone drawn with dust shows some: with few bands the draw can miss.
            keep[np.flatnonzero(eligible)[np.argmin(u[eligible])]] = True
        a, b = (int(x) for x in np.random.default_rng(cells).integers(0, 2**63, size=2))
        return cls(
            dust,
            table,
            e[keep],
            start,
            chalcedony_end,
            SmoothNoise3(volume.solid, dust.patch_mm, np.random.default_rng(patch)),
            SmoothNoise3(volume.solid, dust.cluster_mm, np.random.default_rng(cluster)),
            a,
            b,
        )

    @property
    def pigment(self) -> Pigment:
        return PIGMENTS[self.dust.pigment]

    def deltas(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Distance (mm) into the host band from the dusty boundary on each side of
        each step coordinate x (previous, next); inf where there is none."""
        e = self.dusty
        n = len(x)
        if not len(e):
            return np.full(n, np.inf), np.full(n, np.inf)
        i = np.searchsorted(e, x)
        out: list[FloatArray] = []
        sign = 1.0 if self.dust.side == "start" else -1.0
        for j in (i - 1, i):
            ok = (j >= 0) & (j < len(e))
            ej = e[np.clip(j, 0, len(e) - 1)]
            host = ej if self.dust.side == "start" else ej - 1
            d = sign * (x - ej) * self.table.step_mm[host]
            out.append(np.where(ok, d, np.inf))
        return out[0], out[1]

    def amount(
        self,
        points: FloatArray,
        t: FloatArray,
        chalcedony: BoolArray,
        pixel_mm: float = 0.0,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Cover (0..1) of dust line, specks and blobs at the points, from growth
        time t (smooth, NaN where nothing grew). Only chalcedony takes dust, and
        none of it in the floor or on white bands."""
        n = len(points)
        line, speck, blob = np.zeros(n), np.zeros(n), np.zeros(n)
        x = np.where(np.isfinite(t), t, -1.0) * TIME_STEPS
        k = np.clip(np.floor(x).astype(np.int64), 0, TIME_STEPS - 1)
        ok = chalcedony & np.isfinite(t) & (x >= self.start * TIME_STEPS)
        idx = np.flatnonzero(ok)
        if not len(idx):
            return line, speck, blob
        half = pixel_mm / 2
        d = self.dust
        before, after = self.deltas(x[idx])
        if d.lines:
            cover = np.maximum(
                strip_cover(before, d.line_mm, half), strip_cover(after, d.line_mm, half)
            )
            hit = cover > 0
            if hit.any():
                patch = np.clip(0.6 + 0.6 * self.patch_noise(points[idx[hit]]), 0, 1)
                line[idx[hit]] = cover[hit] * patch
        if d.specks:
            reach = SPECK_MM[1] / SPECK_ELONGATION[0] + half  # thickest speck, both sides
            for delta in (before, after):
                near = np.abs(delta) < reach
                if near.any():
                    j = idx[near]
                    speck[j] = np.maximum(speck[j], self._specks(points[j], delta[near], pixel_mm))
        if d.blobs:
            depth = (t[idx] - self.start) / max(self.chalcedony_end - self.start, 1e-9)
            inner = (depth > INNER) & ~self.table.white[k[idx]]
            if inner.any():
                j = idx[inner]
                blob[j] = self._blobs(points[j], pixel_mm)
        return line, speck, blob

    def _specks(self, p: FloatArray, delta: FloatArray, pixel_mm: float) -> FloatArray:
        """Specks: a ball of the half-length about each candidate, cut to a strip of
        twice the half-thickness on the host side of the boundary, so each is an
        oval drawn out along the band. Present with the cell's fill, thinned by
        cluster noise at the candidate."""
        c = SPECK_CELL_MM
        base = np.floor(p / c).astype(np.int64)
        best = np.zeros(len(p))
        lo_l, hi_l = math.log(SPECK_MM[0] / 2), math.log(SPECK_MM[1] / 2)
        for off in _OFFSETS:
            cell = base + off
            u = cell_uniforms(self.speck_seed, cell, 6)
            q: FloatArray = (cell + u[:, 1:4]) * c
            half_len = np.exp(lo_l + (hi_l - lo_l) * u[:, 4])
            half_thick = half_len / (SPECK_ELONGATION[0] + np.diff(SPECK_ELONGATION)[0] * u[:, 5])
            r: FloatArray = np.sqrt(((p - q) ** 2).sum(axis=1))
            maybe: BoolArray = r < half_len + pixel_mm
            if not maybe.any():
                continue
            fill = self.dust.speck_fill * np.clip(0.5 + 0.7 * self.cluster_noise(q[maybe]), 0, 1)
            present = u[maybe, 0] < fill
            cover = disc_cover(r[maybe], half_len[maybe], pixel_mm) * strip_cover(
                delta[maybe], 2 * half_thick[maybe], pixel_mm / 2
            )
            best[maybe] = np.maximum(best[maybe], np.where(present, cover, 0.0))
        return best

    def _blobs(self, p: FloatArray, pixel_mm: float) -> FloatArray:
        """Rounded aggregates at random: per cm² of face, a density in 3D of
        per_cm2 / 100 / (2 x mean radius) (a section meets a ball of radius R
        within R of the plane)."""
        c = BLOB_CELL_MM
        lo_r, hi_r = math.log(BLOB_MM[0] / 2), math.log(BLOB_MM[1] / 2)
        mean_r = (math.exp(hi_r) - math.exp(lo_r)) / (hi_r - lo_r)
        chance = self.dust.blob_per_cm2 / 100 / (2 * mean_r) * c**3
        base = np.floor(p / c).astype(np.int64)
        best = np.zeros(len(p))
        for off in _OFFSETS:
            cell = base + off
            u = cell_uniforms(self.blob_seed, cell, 5)
            q: FloatArray = (cell + u[:, 1:4]) * c
            radius = np.exp(lo_r + (hi_r - lo_r) * u[:, 4])
            r: FloatArray = np.sqrt(((p - q) ** 2).sum(axis=1))
            cover = np.where(u[:, 0] < chance, disc_cover(r, radius, pixel_mm), 0.0)
            best = np.maximum(best, cover)
        return best


# --- checks (each with a negative control in tests/test_dust.py) ----------------------------


def dust_excess(impurity: FloatArray, band: NDArray[np.integer], banded: BoolArray) -> FloatArray:
    """Impurity above its band's median, read from a cut alone: a thin line or speck
    barely moves the median, so it stands out; a band's own colour does not."""
    out = np.zeros(impurity.shape)
    b = band[banded]
    imp = impurity[banded].astype(np.float64)
    order = np.argsort(b, kind="stable")
    bs, vs = b[order], imp[order]
    starts = np.flatnonzero(np.r_[True, bs[1:] != bs[:-1]])
    ends = np.r_[starts[1:], len(bs)]
    med = np.empty(len(bs))
    for s, e in zip(starts, ends, strict=True):
        med[s:e] = np.median(vs[s:e])
    excess = np.empty(len(bs))
    excess[order] = np.clip(vs - med, 0, None)
    out[banded] = excess
    return out


def boundary_mask(band: NDArray[np.integer], banded: BoolArray, near_px: int) -> BoolArray:
    """Banded pixels within near_px (4-neighbour steps) of a change of band."""
    edge = np.zeros(band.shape, dtype=bool)
    for a, b in (
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
    ):
        diff = banded[a] & banded[b] & (band[a] != band[b])
        edge[a] |= diff
        edge[b] |= diff
    for _ in range(near_px):
        grown = edge.copy()
        grown[1:] |= edge[:-1]
        grown[:-1] |= edge[1:]
        grown[:, 1:] |= edge[:, :-1]
        grown[:, :-1] |= edge[:, 1:]
        edge = grown
    return edge & banded


class DustOnBands(BaseModel):
    """How dust in a cut sits on band boundaries."""

    model_config = ConfigDict(frozen=True)

    dusty_area: float  # share of banded pixels with excess above `least`
    share: float  # share of the excess within near_mm of a boundary
    enrichment: float  # share / the share of banded area that near
    dusty: bool  # enough dust, and it follows the bands


def dust_on_bands(
    excess: FloatArray,
    band: NDArray[np.integer],
    banded: BoolArray,
    spacing_mm: float,
    near_mm: float = 0.1,
    least: float = 0.03,
    least_area: float = 0.002,
    least_enrichment: float = 1.5,
) -> DustOnBands:
    """The check of ADR 0018 s.11: the share of the dust (impurity above its band's
    median) that lies within near_mm of a band boundary, against the share of the
    banded area that does. Dust that follows the bands is enriched there; dust at
    random is not (negative control 1); bands alone leave no excess (control 2)."""
    near = boundary_mask(band, banded, max(1, round(near_mm / spacing_mm)))
    total = float(excess[banded].sum())
    area = float(near[banded].mean()) if banded.any() else 0.0
    share = float(excess[near].sum()) / total if total > 0 else 0.0
    enrichment = share / area if area > 0 else 0.0
    dusty_area = float((excess[banded] > least).mean()) if banded.any() else 0.0
    return DustOnBands(
        dusty_area=dusty_area,
        share=share,
        enrichment=enrichment,
        dusty=dusty_area > least_area and enrichment >= least_enrichment,
    )
