// Camera + standard lights over the polished cut face (ADR 0007). Needs common.cuh, scene.cuh.
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
//   drops the light seen straight through).
//   64 no per-channel phase reweighting (control for the two-lobe check).
//   DISC only: 128 NEE only (weight 1); 256 phase sampling only (weight 1); 512 both
//   strategies at weight 1 (control: counts the light twice).
//   1024 draw scattering from the whole lobes although they are truncated (control).

struct Cam { float s0, t0, pix; int nx, ny, spp; unsigned int seed; float n_in, D;
             int light; float lth, lph, E, Ld; float mubar; int nmat; int nprof; float dt;
             int order; int flags; float lomc; };

__device__ int matid(const Scene& S, const Cam& C, const unsigned short* prof, float x, float y, float z) {
    if (C.flags & 4) return 0;
    float t = 0.f; int c = classify(S, x, y, z, t, C.flags & 1);
    if (c < 2) return c;
    int u = (int)floorf(t / C.dt); u = min(max(u, 0), C.nprof - 1);
    return 2 + prof[u];
}

// Ratio-tracked transmittance (all channels) along a unit direction over length L.
__device__ void ratio(const Scene& S, const Cam& C, const unsigned short* prof, const float* mua,
                      const float* mus, float x, float y, float z, float dx, float dy, float dz,
                      float L, float* T, Rng& r) {
    T[0] = T[1] = T[2] = 1.f;
    if (C.flags & 2) return;
    float s = 0.f;
    for (int it = 0; it < 100000; ++it) {
        s += -logf(r.next()) / C.mubar;
        if (s >= L) return;
        int m = matid(S, C, prof, x + dx * s, y + dy * s, z + dz * s);
        for (int c = 0; c < 3; ++c) T[c] *= 1.f - (mua[3 * m + c] + mus[3 * m + c]) / C.mubar;
        float tm = fmaxf(T[0], fmaxf(T[1], T[2]));
        if (tm < 0.1f) {                                   // shadow roulette (unbiased)
            if (tm <= 0.f || r.next() * 0.1f >= tm) { T[0] = T[1] = T[2] = 0.f; return; }
            for (int c = 0; c < 3; ++c) T[c] *= 0.1f / tm;
        }
    }
}

// Next-event estimate toward one air direction o (unit, pointing at the light, o.n > 0) from
// a real collision at (x, y, z) with material m, path direction d. fac is the light's
// irradiance on a surface facing it (directional: E; disc sample: Omega Lr). With mis, the
// estimate is weighted against phase sampling (balance heuristic); omega is the solid angle
// the light direction was drawn from (in air). Direct path through the face, plus 1 in 4 the
// path via the mirror image in the bottom face (weight 4; flag 16 drops it).
__device__ void nee(const Scene& S, const Cam& C, const unsigned short* prof, const float* mua,
                    const float* mus, const float* ph, int m, const float* w, float x, float y,
                    float z, float dx, float dy, float dz, float n, float ox, float oy, float oz,
                    float fac, bool mis, float omega, float* acc, Rng& r) {
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
    ratio(S, C, prof, mua, mus, x, y, z, lx, ly, lz, d0 / ct, T, r);
    float wl = mis ? pl / (pl + sample_pdf(ph, m, w, cl)) : 1.f;
    for (int c = 0; c < 3; ++c) acc[c] += w[c] * phase_pdf(ph, m, c, cl) * Ein * T[c] * wl;
    if (isfinite(C.D) && Rb > 0.f && !(C.flags & 16) && r.next() < 0.25f) {
        // mirror path: refract in, reflect off the bottom, arrive along
        // omega_r = ct n - (tangent). Shadow legs: down along m = -omega_r, then up along l.
        float mx = -ct * S.n[0] + tx, my = -ct * S.n[1] + ty, mz = -ct * S.n[2] + tz;
        float L1 = (C.D - d0) / ct, T1[3], T2[3];
        ratio(S, C, prof, mua, mus, x, y, z, mx, my, mz, L1, T1, r);
        ratio(S, C, prof, mua, mus, x + mx * L1, y + my * L1, z + mz * L1, lx, ly, lz, C.D / ct, T2, r);
        float cm = mx * dx + my * dy + mz * dz;                // omega_r . (-d)
        float wm = mis ? 0.25f * pl / (0.25f * pl + sample_pdf(ph, m, w, cm)) : 1.f;
        for (int c = 0; c < 3; ++c)
            acc[c] += w[c] * phase_pdf(ph, m, c, cm) * Ein * Rb * 4.f * T1[c] * T2[c] * wm;
    }
}

extern "C" __global__ void render(const Scene* Sp, const Cam* Cp, const unsigned short* prof,
                                  const float* mua, const float* mus, const float* ph,
                                  float* img, unsigned long long* stats) {
    const Scene& S = *Sp; const Cam& C = *Cp;
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
    int events = 0, real = 0;
    for (;;) {
        float depth = -((x - S.o[0]) * S.n[0] + (y - S.o[1]) * S.n[1] + (z - S.o[2]) * S.n[2]);
        float un = -(dx * S.n[0] + dy * S.n[1] + dz * S.n[2]);   // d(depth)/ds
        float db = un < 0.f ? depth / -un : (un > 0.f ? (C.D - depth) / un : __int_as_float(0x7f800000));
        float s = -logf(r.next()) / C.mubar;
        if (s >= db) {
            x += dx * db; y += dy * db; z += dz * db;
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
                    bool mirror = isfinite(C.D) && !(C.flags & 16) && fresnel(ci, n, 1.f) > 0.f;
                    if (mis && real > 0 && ntop == 0 && (nbot == 0 || (nbot == 1 && mirror))) {
                        float pl = n * n * ci / (Om * ca) * (nbot ? 0.25f : 1.f);
                        wp = lastpp / (lastpp + pl);
                    }
                    for (int c = 0; c < 3; ++c) acc[c] += w[c] * n * n * Lr * wp;
                }
            }
            break;
        }
        x += dx * s; y += dy * s; z += dz * s; ++events;
        int m = matid(S, C, prof, x, y, z);
        if (spectral_step(w, mua, mus, m, C.mubar, r)) {
            ++real;
            if (doNee && (C.order == 0 || real == C.order)) {
                if (disc) {                                     // one direction in the cone
                    float q = r.next() * omc, sb = sqrtf(q * (2.f - q)), cb = 1.f - q;
                    float psi = 6.28318531f * r.next(), cp = cosf(psi) * sb, sp = sinf(psi) * sb;
                    nee(S, C, prof, mua, mus, ph, m, w, x, y, z, dx, dy, dz, n,
                        cb * a0 + cp * e10 + sp * e20, cb * a1 + cp * e11 + sp * e21,
                        cb * a2 + cp * e12 + sp * e22, Om * Lr, mis, Om, acc, r);
                } else {
                    nee(S, C, prof, mua, mus, ph, m, w, x, y, z, dx, dy, dz, n, a0, a1, a2, C.E,
                        false, 0.f, acc, r);
                }
            }
            if (C.order != 0 && real >= C.order) break;
            scatter(ph, m, w, dx, dy, dz, r, !(C.flags & 64), hits ? &lastpp : nullptr,
                    C.flags & 1024);
            ntop = nbot = 0;
        }
        if (!roulette(w, w0, r)) break;
        if (events > 200000) break;
    }
    for (int c = 0; c < 3; ++c) atomicAdd(img + 3 * px + c, acc[c] / C.spp);
    atomicAdd(stats, (unsigned long long)events); atomicAdd(stats + 1, (unsigned long long)real);
}
