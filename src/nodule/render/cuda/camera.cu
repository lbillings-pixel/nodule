// Camera + standard lights over the polished cut face (ADR 0007). Needs common.cuh, scene.cuh.
//
// Orthographic camera on the +n side looking along -n; one path per (pixel, sample), box-filtered
// jitter. The path enters through the smooth face with weight T(0)/n^2 (radiance leaving a denser
// medium: L/n^2 is invariant) and walks by null-collision + Kutz spectral tracking, per-material
// two-lobe Henyey-Greenstein phase function per channel (common.cuh). The slab under the face is D mm deep (inf allowed), smooth, air and black
// below.
// Lights:
//   0 DIRECTIONAL: irradiance E (perpendicular to the beam) from polar angle th, azimuth ph.
//     A delta light through a flat smooth face has one refracted direction, so next-event
//     estimation at each real collision is exact: E_in = T(th) E cos th / cos th_t, times the
//     phase function, times the ratio-tracked transmittance to the face (Novak 2014); plus once
//     via the mirror image in the bottom face (factor R_bottom), sampled 1 in 4 with weight 4.
//   1 DOME: uniform radiance Ld over the upper hemisphere; a path escaping up through the face
//     adds w n^2 Ld. The face's own specular reflection F(0) Ld is added on the host.
// Flags (controls and checks):
//   1 no fracture backward map; 2 no ratio tracking (Tr = 1); 4 homogeneous material 0, no
//   geometry; 8 omit 1/n^2 at the camera exit; 16 no bottom-mirror NEE term; 32 dome counts
//   scattered light only (drops the ghost: the dome seen unscattered in the back face).
//   64 no per-channel phase reweighting (control for the two-lobe check).

struct Cam { float s0, t0, pix; int nx, ny, spp; unsigned int seed; float n_in, D;
             int light; float lth, lph, E, Ld; float mubar; int nmat; int nprof; float dt;
             int order; int flags; };

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
    // light, refracted: shadow direction l (toward the face) and factors
    float st = sinf(C.lth) / n, ct = sqrtf(1.f - st * st);
    float ax = cosf(C.lph) * S.u[0] + sinf(C.lph) * S.v[0], ay = cosf(C.lph) * S.u[1] + sinf(C.lph) * S.v[1],
          az = cosf(C.lph) * S.u[2] + sinf(C.lph) * S.v[2];
    float lx = ct * S.n[0] + st * ax, ly = ct * S.n[1] + st * ay, lz = ct * S.n[2] + st * az;
    float Ein = (1.f - fresnel(cosf(C.lth), 1.f, n)) * C.E * cosf(C.lth) / ct;
    float Rb = fresnel(ct, n, 1.f);
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
                continue;
            }
            if (un < 0.f && C.light == 1 && !((C.flags & 32) && real == 0)) {   // up into the dome
                for (int c = 0; c < 3; ++c) acc[c] += w[c] * n * n * C.Ld;
            }
            break;
        }
        x += dx * s; y += dy * s; z += dz * s; ++events;
        int m = matid(S, C, prof, x, y, z);
        if (spectral_step(w, mua, mus, m, C.mubar, r)) {
            ++real;
            if (C.light == 0 && (C.order == 0 || real == C.order)) {
                float cl = lx * dx + ly * dy + lz * dz;             // omega_t . (-d), omega_t = -l
                float d0 = -((x - S.o[0]) * S.n[0] + (y - S.o[1]) * S.n[1] + (z - S.o[2]) * S.n[2]);
                float T[3];
                ratio(S, C, prof, mua, mus, x, y, z, lx, ly, lz, d0 / ct, T, r);
                for (int c = 0; c < 3; ++c) acc[c] += w[c] * phase_pdf(ph, m, c, cl) * Ein * T[c];
                if (isfinite(C.D) && Rb > 0.f && !(C.flags & 16) && r.next() < 0.25f) {
                    // mirror path: refract in, reflect off the bottom, arrive along
                    // omega_r = ct n - st a. Shadow legs: down along m = -omega_r, then up along l.
                    float mx = -ct * S.n[0] + st * ax, my = -ct * S.n[1] + st * ay, mz = -ct * S.n[2] + st * az;
                    float L1 = (C.D - d0) / ct, T1[3], T2[3];
                    ratio(S, C, prof, mua, mus, x, y, z, mx, my, mz, L1, T1, r);
                    ratio(S, C, prof, mua, mus, x + mx * L1, y + my * L1, z + mz * L1, lx, ly, lz, C.D / ct, T2, r);
                    float cm = mx * dx + my * dy + mz * dz;                // omega_r . (-d)
                    for (int c = 0; c < 3; ++c)
                        acc[c] += w[c] * phase_pdf(ph, m, c, cm) * Ein * Rb * 4.f * T1[c] * T2[c];
                }
            }
            if (C.order != 0 && real >= C.order) break;
            scatter(ph, m, w, dx, dy, dz, r, !(C.flags & 64));
        }
        if (!roulette(w, w0, r)) break;
        if (events > 200000) break;
    }
    for (int c = 0; c < 3; ++c) atomicAdd(img + 3 * px + c, acc[c] / C.spp);
    atomicAdd(stats, (unsigned long long)events); atomicAdd(stats + 1, (unsigned long long)real);
}
