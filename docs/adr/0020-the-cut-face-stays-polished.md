# 20. The cut face stays polished (item 20 read and closed)

- Status: accepted (owner, 2026-11-21). Closed without a build: the renderer's face stays
  flat, smooth and polished.
- Date: 2026-11-21
- Builds on: [ADR 0007](0007-renderer-cut-face-monte-carlo.md) (the smooth face),
  [ADR 0018](0018-the-variety-gap-broad-mock-ups.md) section 2 (feature 11, "Glints and
  finish", the owner's lowest), [ADR 0019](0019-rendering-the-variety-layers.md) section 6
  (the look without the face's glare) and section 6.12 step 5 (the sparkle view),
  [ADR 0006](0006-band-counts-as-censored-records.md) section 9 (the 31 faces re-tagged
  sawn).

## Context

Item 20 of the work plan, the last feature item: saw marks and polish level. The renderer
draws one finish, a flat polished face (n 1.54; Fresnel reflection 0.045 at normal
incidence; ADR 0007). The question was whether sawn, ground or wet faces, saw marks or
polishing relief should be drawn as well. Two helper passes read on it (2026-11-20/21;
local reports `notes/n1-reading/item20-optics.md` and `item20-surfaces.md`, which list
every source and grade). Claude re-checked the quotes used below in the local PDFs and did
the arithmetic (scratch scripts, seconds).

## What was found

1. **A face loses its mirror long before it stops being clear to look through.** The
   share of a face's reflection that stays specular falls as exp[-(4π cos θ σ/λ)²], σ
   the RMS roughness at periods longer than the wavelength [@harvey2012, Sec. 1, Eq. 1;
   Sec. 2] (*plausible*: a model, which the authors have "not proven" for arbitrarily
   strong scatter, Sec. 5). At 550 nm and normal incidence half the mirror is gone at
   σ ≈ 36 nm (arithmetic). In transmission the same kind of model gives an exponent
   (n − 1)²/4 = 0.073 times the reflection's at normal incidence [@simonsen2009, Eqs
   29a-b] (*plausible*; arithmetic), so a finely ground face could still look polished in
   the look, which leaves the face's reflection out (*speculative*).
2. **Roughness moves light more than it removes it**: "the amount of integrated reflected
   or transmitted light is much less sensitive to the presence of surface roughness"
   [@simonsen2009, p. 8] (*plausible*). So a dry sawn face's whitish veil is about its
   4.5 % Fresnel reflection spread over a wide lobe (inference). Arithmetic with the GGX
   roughness fitted to ground glass (α 0.39-0.55 [@walter2007, Fig. 12]; glass, so a
   transfer), single scattering, camera at the normal: under the uniform dome a rough face
   reflects about as much as a smooth one (0.036 against 0.045 at α 0.4); under the
   sparkle view's 45° lamp it adds a veil of about 3 % of a white standard under that
   lamp, a polished face none; and the interior seen through it blurs by about 0.14 mm per
   mm of depth (median) (*speculative*).
3. **So in the look a dry sawn face could show only as blur, or as the washed-out look.**
   Its veil is the face reflection the look drops (ADR 0019 s.6); keeping it for sawn
   stones would bring back the washed-out renders the owner rejected on 2026-11-09, plus
   blur.
4. **A liquid film restores most of a polish.** Glycerol (n 1.47) on both faces of
   polyethylene films cut their haze from 18.9 % to 5 % [@simonsen2009, Sec. V.B, p. 12]
   (*verified* for polymer film; a transfer, so an assumption for agate). For water on
   chalcedony the model gives about 85 % less scattering, and Fresnel a veil about 8 times
   fainter (0.54 % against 4.5 %) under a sharp 2 % mirror from the film's top
   (*speculative*, arithmetic).
5. **Saw marks were never measured.** No source read gives the radius, spacing or relief
   of saw marks on any stone. Diamond-saw marks are taken off on the lap with 100 grit,
   then 200 and 600 [@moxon2014, p. 329] (*plausible*; a magazine). That they follow arcs
   of the blade's radius is inference. Under the 45° lamp a face's tilt changes the veil
   by 5-18 % per degree (arithmetic, α 0.55 to 0.1); relief of 1-5 µm over 0.5-2 mm (an
   assumption) tilts it by 0.2-1.8°, so arcs would differ by about one or two grey levels
   on dark stones unless a roughness contrast were invented.
6. **Polishing relief in agate has no measured size.** Diamond abrasives are "the only
   means of sanding both the hard agate and the comparatively soft rhyolite and plume
   without severe undercutting" [@kile2002, p. 266] (*plausible*; a magazine). No heights
   were found.

## Decision (owner, 2026-11-21)

1. **Item 20 is closed: the cut face stays polished.** No finish setting, saw marks or
   polishing relief; the renderer is unchanged.
2. **"Glints" in feature 11 meant the crystal sparkle**, built as the sparkle view (ADR
   0019 s.6.12 step 5). Feature 11 closes with it, and the work plan's feature items end
   here.
3. **The 31 faces re-tagged sawn in ADR 0006 s.9 were photographed wet or oiled**
   (owner). That a wet face looks close to polished (finding 4) would explain why the
   helpers first read them as polished (inference); the tags stay as the owner set them.

## Consequences

- No code, test or render changes. Photo comparisons keep the polished face; wet or oiled
  sawn faces are taken to look close to polished (an assumption, from findings 1 and 4).
- The finish tags (ADR 0004) are not rates of dry sawn faces: at least the 31 were wet or
  oiled.
- harvey2012 and simonsen2009 enter the catalogue; the other sources stay in the reports.

## If it is reopened

- Reasons: a set of dry sawn slabs to match, a measured saw-mark or polish-relief source,
  or the owner's wish.
- First step: a mock-up drawn on the host over existing renders (a veil and a flat blur,
  no kernel change) for the owner's eye. The kernel's direct-light estimates assume a
  flat smooth face (`nee` in `camera.cu`), so a rough face is cheap in the dome look and
  the backlit slice only; roughness 0 must reproduce today's renders exactly.
- Sources to fetch then (details in the reports): Köhler et al. 2021, Fig. 3 (fused-silica
  roughness after each stage; open access); Aydin et al. 2013 (rocks sawn by diamond
  blades); Lekner & Dorf 1988 and Twomey et al. 1986 (wet against dry); Bennett & Porteus
  1961; Sanmartín et al. 2011 (granite gloss); Buijs & Korpel-van Houten 1993; Konstanty
  2002 (stone sawing). They give roughness heights, not the slopes or arc shapes that set
  the look, so a sawn face's strength would be set by eye and recorded as an assumption.
