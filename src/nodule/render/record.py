"""What fills each growth-time step, read from agatesim `Fields` (ADR 0007 s.3).

The first half of the materials mapping (materials.py has the second). In
agatesim's 3D growth model every band property is drawn per band and band =
f(t_dep) (agatesim `growth_fields`), so phase, pigment, impurity and
translucency are functions of the growth step alone. `BandRecord.of` checks
this at every point it is given and refuses what the renderer cannot show.
`GrowthGrid.fill` builds one from the same evaluations that fill the grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from agatesim.fields import Fields, Phase, Pigment
from agatesim.models.growth import TIME_STEPS
from numpy.typing import NDArray

STEPS = TIME_STEPS  # agatesim's growth-time steps; t_dep = (k + 0.5) / STEPS

F32 = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
FIELDS = ("phase", "pigment", "impurity", "translucency")


@dataclass(frozen=True)
class BandRecord:
    """Per growth-time step k (length STEPS): what agatesim deposited then."""

    seen: BoolArray
    phase: NDArray[np.uint8]
    pigment: NDArray[np.uint8]  # Pigment; IRON where agatesim gives none (its convention)
    impurity: F32
    translucency: F32

    @classmethod
    def empty(cls) -> BandRecord:
        return cls(
            seen=np.zeros(STEPS, bool),
            phase=np.zeros(STEPS, np.uint8),
            pigment=np.zeros(STEPS, np.uint8),
            impurity=np.zeros(STEPS, np.float32),
            translucency=np.zeros(STEPS, np.float32),
        )

    @classmethod
    def of(cls, f: Fields, inside: BoolArray) -> BandRecord:
        """From fields at points (any shape) and which lie in the stone. Refuses what
        the renderer cannot show: an unfilled point inside the stone (an open cavity or
        a hollow crystal: no void material yet), an inclusion other than a tube
        (dendrites), and a step whose properties differ between points."""
        if f.t_smooth is None:
            raise ValueError("the model gives no smooth growth time (agatesim ADR 0076 s. 6)")
        tube = inside & (f.phase == Phase.INCLUSION) & np.isnan(f.t_dep)
        body = inside & ~tube
        if np.isnan(f.t_smooth[body]).any():
            raise ValueError(
                "unfilled space inside the stone (open cavity or hollow crystal): "
                "the renderer has no void material yet"
            )
        if (f.phase[body] == Phase.INCLUSION).any():
            raise ValueError("inclusions other than tubes (dendrites) are not rendered yet")
        k = np.clip(np.floor(f.t_dep[body] * STEPS).astype(np.intp), 0, STEPS - 1)
        pigment = f.pigment if f.pigment is not None else np.full(f.phase.shape, Pigment.IRON)
        cols = {
            "phase": f.phase[body],
            "pigment": pigment[body].astype(np.uint8),
            "impurity": f.impurity[body],
            "translucency": f.translucency[body],
        }
        out = cls.empty()
        if not len(k):
            return out
        steps, first = np.unique(k, return_index=True)
        for name, col in cols.items():
            rep = np.zeros(STEPS, col.dtype)
            rep[steps] = col[first]
            if not np.array_equal(col, rep[k]):
                raise ValueError(f"{name} differs within a growth step: not a band field")
            getattr(out, name)[steps] = rep[steps]
        out.seen[steps] = True
        return out

    def merge(self, other: BandRecord) -> BandRecord:
        both, pick = self.seen & other.seen, other.seen & ~self.seen
        merged: dict[str, NDArray[np.generic]] = {"seen": self.seen | other.seen}
        for name in FIELDS:
            a, b = getattr(self, name), getattr(other, name)
            if not np.array_equal(a[both], b[both]):
                raise ValueError(f"records disagree on {name} at a growth step")
            merged[name] = np.where(pick, b, a)
        return BandRecord(**merged)  # pyright: ignore[reportArgumentType]

    def nearest_seen(self) -> NDArray[np.intp]:
        """For each step, the nearest step seen (itself if seen; the earlier on a tie).
        Unseen steps lie only where no grid node fell, so only interpolated t reaches them."""
        idx = np.flatnonzero(self.seen)
        if not len(idx):
            raise ValueError("no growth step seen: the grid misses the stone")
        k = np.arange(STEPS)
        j = np.searchsorted(idx, k)
        lo, hi = idx[np.clip(j - 1, 0, len(idx) - 1)], idx[np.clip(j, 0, len(idx) - 1)]
        return np.where(np.abs(k - lo) <= np.abs(hi - k), lo, hi)
