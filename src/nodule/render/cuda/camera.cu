// Camera + standard lights over the polished cut face (ADR 0007). Needs common.cuh, scene.cuh,
// shift.cuh.
//
// Orthographic camera on the +n side looking along -n; one path per (pixel, sample), box-filtered
// jitter. The path enters through the smooth face with weight T(0)/n^2 (radiance leaving a denser
// medium: L/n^2 is invariant) and walks by null-collision + Kutz spectral tracking, per-material
// two-lobe Henyey-Greenstein phase function per channel (common.cuh). The slab under the face is
// D mm deep (inf allowed), smooth, with air below: black, or the BACK light.
// Lights:
//   0 DIRECTIONAL: irradiance E (perpendicular to the beam) from polar angle th, azimuth ph.
//     A delta light through a flat smooth face has one refracted direction, so next-event
//     estimation at each real collision is exact: E_in = T(th) E cos th / cos th_t, times the
//     phase function, times the ratio-tracked transmittance to the face (Novak 2014); plus once
//     via the mirror image in the bottom face (factor R_bottom), sampled 1 in 4 with weight 4.
//   1 DOME: uniform radiance Ld over the upper hemisphere; a path escaping up through the face
//     adds w n^2 Ld. The face's own specular reflection F(0) Ld is added on the host.
//   2 DISC: uniform radiance Lr over a cone of half-angle alpha about (th, ph), Lr set so the
//     irradiance on a surface facing the disc is E (Lr = E / (pi sin^2 alpha)); alpha -> 0 is
//     DIRECTIONAL. Two strategies, combined by the balance heuristic (Veach 1997 ch. 9): NEE
//     toward one direction drawn uniformly in the cone (as DIRECTIONAL, times the cone's solid
//     angle), and phase sampling: a path leaving the face inside the cone adds w n^2 Lr. Both
//     pdfs are taken per steradian inside the stone (air pdf 1/Omega times n^2 cos th_t / cos th).
//     The grains' g ~ 0.998 lobe is far narrower than the refracted disc: phase sampling covers
//     it, NEE covers the rest. Exit routes NEE cannot make (a top-face reflection, or two
//     bottom-face reflections, since the last real collision) keep phase weight 1.
//   3 BACK: uniform radiance Ld over the lower hemisphere behind a finite slab (L3, ADR 0007
//     7b), the camera side dark; a path escaping down through the back face adds w n^2 Ld.
//     Diffuse, so no NEE and no sparkle; renders keep the full phase function (amendment 4).
// Flags (controls and checks):
//   1 no fracture backward map; 2 no ratio tracking (Tr = 1); 4 homogeneous material 0, no
//   geometry; 8 omit 1/n^2 at the camera exit; 16 no bottom-mirror NEE term; 32 dome and back
//   count scattered light only (dome: drops the ghost seen unscattered in the back face; back:
//   drops the light seen straight through). 2048 black bottom: the back face is index-matched and
//   black, so a path reaching depth D going down is lost, as into the rest of a stone (ADR 0014;
//   no bottom reflection, no mirror NEE term; not with the back light). With a nonzero albedo
//   ba the black face is matte instead: an index-matched Lambertian reflector sending the path
//   back up, cosine-weighted, with weight x ba (the look's grey bottom, ADR 0019; dome only,
//   as no NEE reaches the bottom).
//   64 no per-channel phase reweighting (control for the two-lobe check).
//   DISC only: 128 NEE only (weight 1); 256 phase sampling only (weight 1); 512 both
//   strategies at weight 1 (control: counts the light twice).
//   1024 draw scattering from the whole lobes although they are truncated (control).

struct Cam { float s0, t0, pix; int nx, ny, spp; unsigned int seed; float n_in, D;
             int light; float lth, lph, E, Ld; float mubar; int nmat; int nprof; float dt;
             int order; int flags; float lomc; float ba[3]; };

// What a sample sees: m, the row of its scattering and phase function, and a, its absorption.
// Without a shift a = mua[m]. With one (ADR 0016; shift.cuh), a band row that takes it adds
// (gain - 1) x its pigment's absorption (16b: a lens is more of the band's own pigment), and
// with drift m becomes the translucency table's row nearest the drifted translucency.
// The drift is read from the cut's baked grid when there is one (ADR 0016 s.9), else
// worked out here; the lenses always are, being thinner than the grid.
struct Mat { int m; float a[3]; };

__device__ Mat matid(const Scene& S, const Cam& C, const Shift& H, const unsigned short* prof,
                     const float* mua, float x, float y, float z) {
    Mat M; M.m = 0;
    float t = 0.f;
    if (!(C.flags & 4)) {
        int c = classify(S, x, y, z, t, C.flags & 1);           // x, y, z now restored
        if (c < 2) M.m = c;
        else {
            int u = (int)floorf(t / C.dt); u = min(max(u, 0), C.nprof - 1); M.m = 2 + prof[u];
            // a milky lining's stand-in hollow: the lining without the milk (ADR 0019 s.6.12)
            if (S.voidf != nullptr && grid_at(S, S.voidf, x, y, z) > 0.5f) M.m = 2 + prof[C.nprof - 1];
        }
    }
    for (int c = 0; c < 3; ++c) M.a[c] = mua[3 * M.m + c];
    if (S.absorb != nullptr && M.m >= 2) {                      // the layers (ADR 0019)
        long long n = (long long)S.ns * S.nt * S.nw;
        for (int c = 0; c < 3; ++c) M.a[c] = fmaxf(M.a[c] + grid_at(S, S.absorb + c * n, x, y, z), 0.f);
    }
    if (H.band == nullptr || M.m < 2) return M;
    const float* B = H.band + 4 * M.m;
    if (!(B[0] >= 0.f)) return M;                               // takes no shift
    bool lens = B[1] > 0.f || B[2] > 0.f || B[3] > 0.f, drift = H.ntr > 0;
    if (!lens && !drift) return M;                              // below the iron floor: no lens
    float gain, change;
    shift_at(H, x, y, z, t, lens, drift && H.dgrid == nullptr, gain, change);
    if (drift && H.dgrid != nullptr) change = grid_at(S, H.dgrid, x, y, z);  // baked (s.9)
    for (int c = 0; c < 3; ++c) M.a[c] += (gain - 1.f) * B[1 + c];
    if (drift) {
        float tr = fminf(fmaxf(B[0] + change, 0.f), 1.f);
        M.m = H.tr0 + min((int)(tr * (H.ntr - 1) + 0.5f), H.ntr - 1);
    }
    return M;
}

// Ratio-tracked transmittance (all channels) along a unit direction over length L.
__device__ void ratio(const Scene& S, const Cam& C, const Shift& H, const unsigned short* prof,
                      const float* mua, const float* mus, float x, float y, float z, float dx,
                      float dy, float dz, float L, float* T, Rng& r) {
    T[0] = T[1] = T[2] = 1.f;
    if (C.flags & 2) return;
    float s = 0.f;
    for (int it = 0; it < 100000; ++it) {
        s += -logf(r.next()) / C.mubar;
        if (s >= L) return;
        Mat M = matid(S, C, H, prof, mua, x + dx * s, y + dy * s, z + dz * s);
        for (int c = 0; c < 3; ++c) T[c] *= 1.f - (M.a[c] + mus[3 * M.m + c]) / C.mubar;
        float tm = fmaxf(T[0], fmaxf(T[1], T[2]));
        if (tm < 0.1f) {                                   // shadow roulette (unbiased)
            if (tm <= 0.f || r.next() * 0.1f >= tm) { T[0] = T[1] = T[2] = 0.f; return; }
            for (int c = 0; c < 3; ++c) T[c] *= 0.1f / tm;
        }
    }
}

#ifdef NODULE_VOID
#define LEG_SKIP_CAP 4.f   // a shadow leg's longest skip, in steps (vstep)
// Is a straight shadow leg from (x, y, z) along unit l, of length L, blocked by the hollow: does
// it meet the hollow's air, or (a stone's box) leave the shell below the slab (depth D), where
// a path meets the bottom's rule (lost when black)? Air is tested at the flights' steps (vstep)
// from the leg's start, clipped to the box (kind 1), as air lies nowhere else. A shadow leg
// cannot bend at the wall, so one that meets air carries nothing; routes through the hollow
// are left to phase sampling at weight 1 (ADR 0019 s.6.12, the small lamp).
__device__ bool leg_meets_air(const Void& V, const Scene& S, float x, float y, float z,
                              float lx, float ly, float lz, float L, float D) {
    if (V.kind == 0 || !(L > 0.f) || (V.flags & VFLAG_LEGS_IGNORE_AIR)) return false;
    float st = vstep(V), v[2], s0 = 0.f, s1 = L;
    if (V.kind == 1 && vdepth(S, x, y, z) > D) {                // below the slab: in the shell?
        float up = lx * S.n[0] + ly * S.n[1] + lz * S.n[2];     // depth falls at this rate
        float below = up > 0.f ? fminf((vdepth(S, x, y, z) - D) / up, L) : L;
        for (int it = 1; it < 1000000; ++it) {
            float q = fminf(it * st, below);
            if (!in_shell(V, x + lx * q, y + ly * q, z + lz * q)) return true;
            if (q >= below) break;
        }
    }
    if (V.kind == 1) {                                          // the part inside the box
        const float p[3] = {x, y, z}, d[3] = {lx, ly, lz};
        const int a0[3] = {V.ax, V.ay, V.az}, n[3] = {V.nx, V.ny, V.nz};
        for (int a = 0; a < 3; ++a) {
            float lo = a0[a] * V.h, hi = (a0[a] + n[a] - 1) * V.h;
            if (fabsf(d[a]) < 1e-12f) {
                if (p[a] < lo || p[a] >= hi) return false;
                continue;
            }
            float t0 = (lo - p[a]) / d[a], t1 = (hi - p[a]) / d[a];
            s0 = fmaxf(s0, fminf(t0, t1)); s1 = fminf(s1, fmaxf(t0, t1));
        }
        if (!(s0 <= s1)) return false;
    }
    // Skips (sphere tracing on the air test's own values; LEG_SKIP_CAP steps at most). phi, the
    // chalcedony's level, changes by at most ~1.2 per mm (a wall distance plus its wobble), so a
    // point at phi < 0 lies at least -phi / 1.5 from air (tube cores read minus the reach, a
    // jump the cap bounds). The crystals' arrival time changes by at most K = max(elongation,
    // 1 + cot) per mm, and the box's earliest of a cell's 8 corner crystals overstates the true
    // earliest by at most K sqrt(3) h (each corner's owner is the true earliest at its node), so
    // a point at box level -L lies at least L / K - sqrt(3) h from box air (the 2026-11-20
    // review's bound; analytic shapes: h 0, K 1, their level an exact distance).
    float K = fmaxf(V.elong, V.cot_a + 1.f), slack = 1.7320508f * V.h, s = fmaxf(s0, 0.f) + st;
    for (int it = 0; it < 1000000; ++it) {
        float q = fminf(s, s1);
        if (air_at(V, S, x + lx * q, y + ly * q, z + lz * q, v)) return true;
        if (q >= s1) break;
        float skip = 0.f;
        if (v[1] < 0.f) skip = -v[1] / 1.5f;
        else if (v[0] < 0.f) skip = -v[0] / K - slack;
        s += fminf(fmaxf(skip, st), LEG_SKIP_CAP * st);
    }
    return false;
}
#define VOID_PARAM , const Void& V
#define VOID_ARG , V
#else
#define VOID_PARAM
#define VOID_ARG
#endif

// Next-event estimate toward one air direction o (unit, pointing at the light, o.n > 0) from
// a real collision at (x, y, z) with material m, path direction d. fac is the light's
// irradiance on a surface facing it (directional: E; disc sample: Omega Lr). With mis, the
// estimate is weighted against phase sampling (balance heuristic); omega is the solid angle
// the light direction was drawn from (in air). Direct path through the face, plus 1 in 4 the
// path via the mirror image in the bottom face (weight 4; flag 16 drops it). With the hollow
// as air, a leg that meets air carries nothing (`leg_meets_air`).
__device__ void nee(const Scene& S, const Cam& C, const Shift& H, const unsigned short* prof,
                    const float* mua, const float* mus, const float* ph, int m, const float* w, float x, float y,
                    float z, float dx, float dy, float dz, float n, float ox, float oy, float oz,
                    float fac, bool mis, float omega, float* acc, Rng& r VOID_PARAM) {
    float cth = ox * S.n[0] + oy * S.n[1] + oz * S.n[2];
    if (!(cth > 0.f)) return;                                   // below the horizon
    float tx = (ox - cth * S.n[0]) / n, ty = (oy - cth * S.n[1]) / n, tz = (oz - cth * S.n[2]) / n;
    float ct = sqrtf(fmaxf(0.f, 1.f - (tx * tx + ty * ty + tz * tz)));
    float lx = ct * S.n[0] + tx, ly = ct * S.n[1] + ty, lz = ct * S.n[2] + tz;   // toward the face
    float Ein = (1.f - fresnel(cth, 1.f, n)) * fac * cth / ct;
    float pl = mis ? n * n * ct / (omega * cth) : 0.f;          // light pdf, inside solid angle
    float Rb = fresnel(ct, n, 1.f);
    float cl = lx * dx + ly * dy + lz * dz;                     // omega_t . (-d), omega_t = -l
    float d0 = -((x - S.o[0]) * S.n[0] + (y - S.o[1]) * S.n[1] + (z - S.o[2]) * S.n[2]);
    float T[3];
#ifdef NODULE_VOID
    bool open = !leg_meets_air(V, S, x, y, z, lx, ly, lz, d0 / ct, C.D);
#else
    bool open = true;
#endif
    if (open) {
        ratio(S, C, H, prof, mua, mus, x, y, z, lx, ly, lz, d0 / ct, T, r);
        float wl = mis ? pl / (pl + sample_pdf(ph, m, w, cl)) : 1.f;
        for (int c = 0; c < 3; ++c) acc[c] += w[c] * phase_pdf(ph, m, c, cl) * Ein * T[c] * wl;
    }
    if (isfinite(C.D) && Rb > 0.f && !(C.flags & (16 | 2048)) && r.next() < 0.25f) {
        // mirror path: refract in, reflect off the bottom, arrive along
        // omega_r = ct n - (tangent). Shadow legs: down along m = -omega_r, then up along l.
        float mx = -ct * S.n[0] + tx, my = -ct * S.n[1] + ty, mz = -ct * S.n[2] + tz;
        float L1 = (C.D - d0) / ct, T1[3], T2[3];
#ifdef NODULE_VOID
        float bx = x + mx * L1, by = y + my * L1, bz = z + mz * L1;
        if (leg_meets_air(V, S, x, y, z, mx, my, mz, L1, C.D) ||
            leg_meets_air(V, S, bx, by, bz, lx, ly, lz, C.D / ct, C.D)) return;
#endif
        ratio(S, C, H, prof, mua, mus, x, y, z, mx, my, mz, L1, T1, r);
        ratio(S, C, H, prof, mua, mus, x + mx * L1, y + my * L1, z + mz * L1, lx, ly, lz, C.D / ct, T2, r);
        float cm = mx * dx + my * dy + mz * dz;                // omega_r . (-d)
        float wm = mis ? 0.25f * pl / (0.25f * pl + sample_pdf(ph, m, w, cm)) : 1.f;
        for (int c = 0; c < 3; ++c)
            acc[c] += w[c] * phase_pdf(ph, m, c, cm) * Ein * Rb * 4.f * T1[c] * T2[c] * wm;
    }
}

// The hollow as air (NODULE_VOID, a separate compile: voidbox.cuh; ADR 0019 s.6.12 step 4).
// Under the dome, or the small lamp (disc: the sparkle view, step 5). No shadow ray crosses the
// wall: one that meets air carries nothing (`leg_meets_air`), and a path that met the wall
// since its last real collision keeps phase weight 1, as does one leaving through the open
// mouth into the lamp (no face there, so no n^2). The path carries its medium. A free flight in the
// stone is tested for air at steps of at most vstep, then bisected; at the wall it reflects or
// refracts by Fresnel (n to 1, normal from the level or phi), weight x n^2 into the air. In air
// it runs straight, tested at the same steps, to the next wall (/ n^2 back into the stone) or up
// through the face's plane at the open mouth (adds w Ld: no face there). A camera ray over an
// open pixel starts in air at weight 1. Below the slab (depth D) the bottom does not apply
// within the void box's shell; a path leaving the shell below D meets the bottom's rule there
// (black: lost; matte: sent back up once, from inside the shell; ADR 0019 s.6.12, owner's
// option (b), *assumption*). stats[2], [3], [4]: wall meetings, those the gate alone decided,
// and those that passed through the wall (the rest reflected); [5], [6]: the bottom rule's
// turns at the shell's exit, and the exits.
extern "C" __global__ void render(const Scene* Sp, const Cam* Cp, const Shift* Hp,
#ifdef NODULE_VOID
                                  const Void* Vp,
#endif
                                  const unsigned short* prof, const float* mua, const float* mus,
                                  const float* ph, float* img, unsigned long long* stats) {
    const Scene& S = *Sp; const Cam& C = *Cp; const Shift& H = *Hp;
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long npx = (long long)C.nx * C.ny;
    if (id >= npx * C.spp) return;
    long long px = id % npx; int ix = (int)(px % C.nx), iy = (int)(px / C.nx);
    Rng r; r.init(C.seed, id);
    float n = C.n_in;
    float ps = C.s0 + (ix + r.next()) * C.pix, pt = C.t0 + (iy + r.next()) * C.pix;
    float x = S.o[0] + ps * S.u[0] + pt * S.v[0], y = S.o[1] + ps * S.u[1] + pt * S.v[1],
          z = S.o[2] + ps * S.u[2] + pt * S.v[2];
    float dx = -S.n[0], dy = -S.n[1], dz = -S.n[2];            // into the stone
    float w0 = (1.f - fresnel(1.f, 1.f, n)) / ((C.flags & 8) ? 1.f : n * n);
    float w[3] = {w0, w0, w0}, acc[3] = {0.f, 0.f, 0.f};
    // light axis a (toward the light, in air) and a frame e1, e2 around it
    float cP = cosf(C.lph), sP = sinf(C.lph), cA = cosf(C.lth), sA = sinf(C.lth);
    float h0 = cP * S.u[0] + sP * S.v[0], h1 = cP * S.u[1] + sP * S.v[1], h2 = cP * S.u[2] + sP * S.v[2];
    float a0 = cA * S.n[0] + sA * h0, a1 = cA * S.n[1] + sA * h1, a2 = cA * S.n[2] + sA * h2;
    float e10 = -sA * S.n[0] + cA * h0, e11 = -sA * S.n[1] + cA * h1, e12 = -sA * S.n[2] + cA * h2;
    float e20 = -sP * S.u[0] + cP * S.v[0], e21 = -sP * S.u[1] + cP * S.v[1], e22 = -sP * S.u[2] + cP * S.v[2];
    // disc: omc = 1 - cos(alpha) from the host (accurate for small alpha), solid angle Om
    bool disc = C.light == 2;
    float omc = C.lomc, Om = 6.28318531f * omc;
    float Lr = disc ? C.E / (3.14159265f * omc * (2.f - omc)) : 0.f;
    bool hits = disc && C.order == 0 && !(C.flags & 128);       // phase-sampled lamp hits
    bool doNee = C.light == 0 || (disc && !(C.flags & 256));
    bool mis = disc && C.order == 0 && !(C.flags & (128 | 256 | 512));
    float lastpp = 0.f; int ntop = 0, nbot = 0;                 // since the last real collision
    bool walled = false;                                        // ditto: met the hollow's wall
    int events = 0, real = 0;
#ifdef NODULE_VOID
    const Void& V = *Vp;
    const float INF = __int_as_float(0x7f800000);
    const float wn2 = (V.flags & VFLAG_NO_WALL_N2) ? 1.f : n * n;
    unsigned long long crossings = 0, guessed = 0, turned = 0, shellx = 0, exits = 0;
    bool inair = false, outside = false;
    {
        float v[2], e = 1e-4f;
        if (air_at(V, S, x + dx * e, y + dy * e, z + dz * e, v)) {   // an open pixel: no face
            inair = true; w0 = 1.f; w[0] = w[1] = w[2] = 1.f;
        }
    }
#endif
    for (;;) {
        float depth = -((x - S.o[0]) * S.n[0] + (y - S.o[1]) * S.n[1] + (z - S.o[2]) * S.n[2]);
        float un = -(dx * S.n[0] + dy * S.n[1] + dz * S.n[2]);   // d(depth)/ds
        float db = un < 0.f ? depth / -un : (un > 0.f ? (C.D - depth) / un : __int_as_float(0x7f800000));
#ifdef NODULE_VOID
        if (inair) {                                            // straight to the next wall
            float dface = un < 0.f ? depth / -un : INF, st = vstep(V), s0 = 0.f, s1 = 0.f, v[2];
            int hit = 0;                                        // 1 a wall, 2 the open face
            for (int it = 0; it < 100000 && !hit; ++it) {
                s1 = s0 + st;
                if (s1 >= dface) {                              // the face's plane first: test
                    s1 = fmaxf(s0, dface - 1e-4f);              // just under it, then leave
                    if (air_at(V, S, x + dx * s1, y + dy * s1, z + dz * s1, v)) { hit = 2; break; }
                }
                if (!air_at(V, S, x + dx * s1, y + dy * s1, z + dz * s1, v)) hit = 1;
                else s0 = s1;
            }
            if (hit == 2) {                                     // up out of the mouth: the dome
                if (C.light == 1 && !((C.flags & 32) && real == 0))
                    for (int c = 0; c < 3; ++c) acc[c] += w[c] * C.Ld;
                if (hits && dx * a0 + dy * a1 + dz * a2 >= 1.f - omc)   // or into the lamp
                    for (int c = 0; c < 3; ++c) acc[c] += w[c] * Lr;
                break;
            }
            if (hit == 0) break;
            float sa = s0, ss = s1, vs[2] = {v[0], v[1]}, m[3];
            bisect(V, S, x, y, z, dx, dy, dz, sa, ss, vs);
            float xa = x + dx * sa, ya = y + dy * sa, za = z + dz * sa;
            float xs = x + dx * ss, ys = y + dy * ss, zs = z + dz * ss;
            ++crossings; if (!wall_normal(V, S, xa, ya, za, vs, m)) ++guessed;
            if (!(dx * m[0] + dy * m[1] + dz * m[2] < -VOID_FACING)) {      // see VOID_FACING
                m[0] = -dx; m[1] = -dy; m[2] = -dz; ++guessed;
            }
            float d[3] = {dx, dy, dz};
            if (fresnel_turn(d, m, 1.f, n, r.next())) {         // into the stone
                ++turned; x = xs; y = ys; z = zs; inair = false;
                for (int c = 0; c < 3; ++c) w[c] /= wn2;
            } else { x = xa; y = ya; z = za; }
            dx = d[0]; dy = d[1]; dz = d[2];
            ntop = nbot = 0; walled = true;
            if (++events > 200000) break;
            continue;
        }
        // below the slab only through the hollow's air: a stone path from above meets the
        // bottom at D as without a hollow; one already below (it came through the air, to the
        // far wall) has no bottom there (owner's option (b) as refined 2026-11-19)
        if (un > 0.f && V.kind == 1 && depth >= C.D) db = INF;
#endif
        float s = -logf(r.next()) / C.mubar;
#ifdef NODULE_VOID
        if (V.kind != 0) {                                      // does the flight meet air?
            float seg = fminf(s, db), st = vstep(V), s0 = 0.f, s1 = 0.f, v[2];
            bool hit = false;
            while (s0 < seg && !hit) {
                s1 = fminf(s0 + st, seg);
                if (air_at(V, S, x + dx * s1, y + dy * s1, z + dz * s1, v)) hit = true;
                else s0 = s1;
            }
            if (hit) {
                float sa = s1, ss = s0, vs[2], m[3];
                air_at(V, S, x + dx * ss, y + dy * ss, z + dz * ss, vs);
                bisect(V, S, x, y, z, dx, dy, dz, sa, ss, vs);
                float xa = x + dx * sa, ya = y + dy * sa, za = z + dz * sa;
                float xs = x + dx * ss, ys = y + dy * ss, zs = z + dz * ss;
                ++crossings; if (!wall_normal(V, S, xa, ya, za, vs, m)) ++guessed;
                if (!(dx * m[0] + dy * m[1] + dz * m[2] > VOID_FACING)) {   // see VOID_FACING
                    m[0] = dx; m[1] = dy; m[2] = dz; ++guessed;
                }
                float d[3] = {dx, dy, dz}, nn[3] = {-m[0], -m[1], -m[2]};   // facing the stone
                if (fresnel_turn(d, nn, n, 1.f, r.next())) {    // out into the air
                    ++turned; x = xa; y = ya; z = za; inair = true;
                    for (int c = 0; c < 3; ++c) w[c] *= wn2;
                } else { x = xs; y = ys; z = zs; }
                dx = d[0]; dy = d[1]; dz = d[2];
                ntop = nbot = 0; walled = true;
                if (++events > 200000) break;
                continue;
            }
        }
#endif
        if (s >= db) {
            x += dx * db; y += dy * db; z += dz * db;
            if (un > 0.f && (C.flags & 2048)) {                 // black or matte bottom
                if (C.ba[0] <= 0.f && C.ba[1] <= 0.f && C.ba[2] <= 0.f) break;   // black: lost
                // matte: index-matched Lambertian, albedo ba per channel, cosine-weighted up
                for (int c = 0; c < 3; ++c) w[c] *= C.ba[c];
                float ct = sqrtf(1.f - r.next()), st = sqrtf(fmaxf(0.f, 1.f - ct * ct));
                float p = 6.28318531f * r.next(), cu = cosf(p) * st, sv = sinf(p) * st;
                dx = ct * S.n[0] + cu * S.u[0] + sv * S.v[0];
                dy = ct * S.n[1] + cu * S.u[1] + sv * S.v[1];
                dz = ct * S.n[2] + cu * S.u[2] + sv * S.v[2];
                ntop = nbot = 0;
                if (!roulette(w, w0, r)) break;
                continue;
            }
            if (r.next() <= fresnel(fabsf(un), n, 1.f)) {       // internal reflection
                float k = 2.f * (dx * S.n[0] + dy * S.n[1] + dz * S.n[2]);
                dx -= k * S.n[0]; dy -= k * S.n[1]; dz -= k * S.n[2];
                if (un < 0.f) ++ntop; else ++nbot;
                continue;
            }
            if (un < 0.f && C.light == 1 && !((C.flags & 32) && real == 0)) {   // up into the dome
                for (int c = 0; c < 3; ++c) acc[c] += w[c] * n * n * C.Ld;
            }
            if (un > 0.f && C.light == 3 && !((C.flags & 32) && real == 0)) {   // down into the backlight
                for (int c = 0; c < 3; ++c) acc[c] += w[c] * n * n * C.Ld;
            }
            if (un < 0.f && hits) {                             // up: into the disc?
                float ci = -un, ca = sqrtf(fmaxf(0.f, 1.f - n * n * (1.f - ci * ci)));
                float ox = ca * S.n[0] + n * (dx - ci * S.n[0]), oy = ca * S.n[1] + n * (dy - ci * S.n[1]),
                      oz = ca * S.n[2] + n * (dz - ci * S.n[2]);
                if (ox * a0 + oy * a1 + oz * a2 >= 1.f - omc) {
                    float wp = 1.f;                             // routes NEE cannot make: 1
                    bool mirror = isfinite(C.D) && !(C.flags & (16 | 2048)) && fresnel(ci, n, 1.f) > 0.f;
                    if (mis && real > 0 && !walled && ntop == 0 && (nbot == 0 || (nbot == 1 && mirror))) {
                        float pl = n * n * ci / (Om * ca) * (nbot ? 0.25f : 1.f);
                        wp = lastpp / (lastpp + pl);
                    }
                    for (int c = 0; c < 3; ++c) acc[c] += w[c] * n * n * Lr * wp;
                }
            }
            break;
        }
#ifdef NODULE_VOID
        float xp = x, yp = y, zp = z;                           // before the flight
#endif
        x += dx * s; y += dy * s; z += dz * s; ++events;
#ifdef NODULE_VOID
        // left the shell below the slab: the slab's bottom there too (owner's option (b),
        // 2026-11-19): lost when black; when matte, sent back up as from the bottom, ONCE: the
        // turn is made from where the flight began, inside the shell (2026-11-19; left
        // outside, the next short flight met the rule again, 2.3 turns an exit on 4/4).
        // stats[5], [6]: such turns, and the exits they belong to (a run of turns with no
        // other event between is one exit). Control VFLAG_SHELL_STAY: the old turn in place.
        if (V.kind == 1 && vdepth(S, x, y, z) > C.D && !in_shell(V, x, y, z)) {
            ++shellx; if (!outside) ++exits; outside = true;
            if (C.ba[0] <= 0.f && C.ba[1] <= 0.f && C.ba[2] <= 0.f) break;
            for (int c = 0; c < 3; ++c) w[c] *= C.ba[c];
            if (!(V.flags & VFLAG_SHELL_STAY)) { x = xp; y = yp; z = zp; }
            float ct = sqrtf(1.f - r.next()), st = sqrtf(fmaxf(0.f, 1.f - ct * ct));
            float p = 6.28318531f * r.next(), cu = cosf(p) * st, sv = sinf(p) * st;
            dx = ct * S.n[0] + cu * S.u[0] + sv * S.v[0];
            dy = ct * S.n[1] + cu * S.u[1] + sv * S.v[1];
            dz = ct * S.n[2] + cu * S.u[2] + sv * S.v[2];
            ntop = nbot = 0;
            if (!roulette(w, w0, r)) break;
            continue;
        }
        outside = false;
#endif
        Mat M = matid(S, C, H, prof, mua, x, y, z);
        int m = M.m;
        if (spectral_step(w, M.a, mus, m, C.mubar, r)) {
            ++real;
            if (doNee && (C.order == 0 || real == C.order)) {
                if (disc) {                                     // one direction in the cone
                    float q = r.next() * omc, sb = sqrtf(q * (2.f - q)), cb = 1.f - q;
                    float psi = 6.28318531f * r.next(), cp = cosf(psi) * sb, sp = sinf(psi) * sb;
                    nee(S, C, H, prof, mua, mus, ph, m, w, x, y, z, dx, dy, dz, n,
                        cb * a0 + cp * e10 + sp * e20, cb * a1 + cp * e11 + sp * e21,
                        cb * a2 + cp * e12 + sp * e22, Om * Lr, mis, Om, acc, r VOID_ARG);
                } else {
                    nee(S, C, H, prof, mua, mus, ph, m, w, x, y, z, dx, dy, dz, n, a0, a1, a2, C.E,
                        false, 0.f, acc, r VOID_ARG);
                }
            }
            if (C.order != 0 && real >= C.order) break;
            scatter(ph, m, w, dx, dy, dz, r, !(C.flags & 64), hits ? &lastpp : nullptr,
                    C.flags & 1024);
            ntop = nbot = 0; walled = false;
        }
        if (!roulette(w, w0, r)) break;
        if (events > 200000) break;
    }
    for (int c = 0; c < 3; ++c) atomicAdd(img + 3 * px + c, acc[c] / C.spp);
    atomicAdd(stats, (unsigned long long)events); atomicAdd(stats + 1, (unsigned long long)real);
#ifdef NODULE_VOID
    if (crossings) { atomicAdd(stats + 2, crossings); atomicAdd(stats + 3, guessed);
                     atomicAdd(stats + 4, turned); }
    if (shellx) { atomicAdd(stats + 5, shellx); atomicAdd(stats + 6, exits); }
#endif
}
