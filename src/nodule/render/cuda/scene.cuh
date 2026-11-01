// The specimen on the GPU (ADR 0007 section 1): what lies at a PRESENT-space world point (mm).
// classify: 0 outside the stone, 1 tube core, 2 stone (t = smooth growth time from the grid).
// The fractures are undone exactly, mirroring agatesim volume.fractures.Fracture.restore
// (explicit steps of -tau v, BELL profiles, youngest first); the stone (sphere-union level > 0)
// and tube (segment distance <= radius) tests are exact at the restored point; only t comes
// from the grid (trilinear, restored coordinates, plane frame). tests/test_render_gpu.py checks
// this against Specimen.evaluate, so a change in agatesim's fracture map shows up there.

struct Scene {
    float o[3], u[3], v[3], n[3];          // plane: origin, u, v, normal (stone at (p-o).n < 0)
    float s0, t0, w0, h; int ns, nt, nw;    // grid axes (restored coords, plane frame)
    int nsph, ntube, nfrac;
    const float* sph;   // nsph x 4: centre, radius
    const float* tube;  // ntube x 7: a, b, radius
    const float* frac;  // nfrac x 17: centre 3, frame rows e0 e1 e2 (9), slip, ra, rc, drag, steps
    const float* grid;  // (nw, nt, ns), NaN cells already filled
};

__device__ __forceinline__ float bell(float s) { float c = fminf(s, 1.f); float q = 1.f - c * c; return q * q; }

__device__ void restore(const Scene& S, float& x, float& y, float& z) {
    for (int f = S.nfrac - 1; f >= 0; --f) {            // youngest first
        const float* F = S.frac + 17 * f;
        float dx = x - F[0], dy = y - F[1], dz = z - F[2];
        float g0 = F[3] * dx + F[4] * dy + F[5] * dz;
        if (!(g0 > 0.f)) continue;
        float g1 = F[6] * dx + F[7] * dy + F[8] * dz, g2 = F[9] * dx + F[10] * dy + F[11] * dz;
        float slip = F[12], ra = F[13], rc = F[14], drag = F[15]; int steps = (int)F[16];
        float across = isinf(drag) ? 1.f : bell(fabsf(g0) / drag);
        float tau = 1.f / steps, g = g1;
        for (int k = 0; k < steps; ++k) {
            float r = sqrtf((g / ra) * (g / ra) + (g2 / rc) * (g2 / rc));
            g -= tau * slip * bell(r) * across;
        }
        float d = g - g1;
        x += d * F[6]; y += d * F[7]; z += d * F[8];
    }
}

__device__ float tlookup(const Scene& S, float x, float y, float z) {
    float rx = x - S.o[0], ry = y - S.o[1], rz = z - S.o[2];
    float fs = ((rx * S.u[0] + ry * S.u[1] + rz * S.u[2]) - S.s0) / S.h;
    float ft = ((rx * S.v[0] + ry * S.v[1] + rz * S.v[2]) - S.t0) / S.h;
    float fw = (-(rx * S.n[0] + ry * S.n[1] + rz * S.n[2]) - S.w0) / S.h;
    fs = fminf(fmaxf(fs, 0.f), S.ns - 1.001f); ft = fminf(fmaxf(ft, 0.f), S.nt - 1.001f);
    fw = fminf(fmaxf(fw, 0.f), S.nw - 1.001f);
    int i = (int)fs, j = (int)ft, k = (int)fw; float a = fs - i, b = ft - j, c = fw - k;
    long long sy = S.ns, sz = (long long)S.ns * S.nt;
    const float* q = S.grid + k * sz + j * sy + i;
    return (1-c)*((1-b)*((1-a)*q[0] + a*q[1]) + b*((1-a)*q[sy] + a*q[sy+1]))
         +    c *((1-b)*((1-a)*q[sz] + a*q[sz+1]) + b*((1-a)*q[sz+sy] + a*q[sz+sy+1]));
}

// mode bit 1: skip the fracture backward map (negative control)
__device__ int classify(const Scene& S, float x, float y, float z, float& t, int mode) {
    if (!(mode & 1)) restore(S, x, y, z);
    float lev = -1e30f;
    for (int i = 0; i < S.nsph; ++i) {
        const float* c = S.sph + 4 * i;
        float dx = x - c[0], dy = y - c[1], dz = z - c[2];
        lev = fmaxf(lev, c[3] - sqrtf(dx * dx + dy * dy + dz * dz));
    }
    if (!(lev > 0.f)) return 0;
    for (int i = 0; i < S.ntube; ++i) {
        const float* T = S.tube + 7 * i;
        float ax = T[3] - T[0], ay = T[4] - T[1], az = T[5] - T[2];
        float px = x - T[0], py = y - T[1], pz = z - T[2];
        float l2 = ax * ax + ay * ay + az * az;
        float s = l2 > 0.f ? fminf(fmaxf((px * ax + py * ay + pz * az) / l2, 0.f), 1.f) : 0.f;
        px -= s * ax; py -= s * ay; pz -= s * az;
        if (px * px + py * py + pz * pz <= T[6] * T[6]) return 1;
    }
    t = tlookup(S, x, y, z);
    return 2;
}

extern "C" __global__ void classify_points(long long n, const float* pts, const Scene* Sp,
                                           int mode, int* cls, float* tout) {
    const Scene& S = *Sp;
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    float t = __int_as_float(0x7fc00000);
    cls[p] = classify(S, pts[3 * p], pts[3 * p + 1], pts[3 * p + 2], t, mode);
    tout[p] = t;
}
