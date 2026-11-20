// The hollow drawn as air (ADR 0019 s.6.12, step 4). Needs scene.cuh and agatesim's
// CRYSTAL_TIME_CUDA (agatesim ADR 0090), prepended by the host, never copied here.
//
// A Void says where the stone is air. kind 1 is the stone's void box (nodule.render.voidbox):
// at world nodes (a + index) h, each node's crystal id and flags (1 gated, 2 chalcedony near
// its last surface, 4 within the shell); air = all 8 corners of the point's cell flagged 1 or 2,
// the earliest of their 8 crystals later than the stage's last reach (level > 0) and, with phi,
// the chalcedony's last surface passed (phi > 0, phi read trilinearly at h / 2). Outside the
// box: not air. kinds 2, 3 and 4 are analytic shapes for the checks: an air layer between depths
// p[0] < p[1] below the face, an air sphere of centre p[0..2] and radius p[3], and a flat facet:
// air below the face on the side of the plane through p[0..2] that its unit normal p[3..5]
// points to (open at the face; the small lamp's glint check).
// Both values (level, phi) are positive in air; their gradients point into the air.

struct Void {
    int kind, flags, nx, ny, nz, ax, ay, az;
    int pnx, pny, pnz, faces;   // faces: the crystals' (agatesim ADR 0092; 0 round)
    float h, cot_a, elong, last;
    float p[8];
    const int* ids; const unsigned char* nf; const float* phi; const float* rows;
};

#define VOID_GATE 3      // gated, or chalcedony near its last surface
#define VOID_SHELL 4
#define VFLAG_NO_WALL_N2 1   // control: leave the n^2 out at the hollow's wall
#define VFLAG_SHELL_STAY 2   // control: a shell exit below the slab turns outside (pre-fix)
#define VFLAG_LEGS_IGNORE_AIR 4   // control: shadow legs pass the hollow as if it were stone
// A crossing into air needs a normal facing along the path (d . m > 0; into the stone < 0).
// Where the found normal does not (|cos| under this, or the wrong side: the box's level jumps
// where a cell's corner crystals change, s.6.12 step 1, so the wall there is a cell face, not
// that crystal's), the wall is taken as met head-on (m = +-d) and the normal counted as guessed;
// else such a path reflects into itself and meets the same wall again until the event cap
// (found on the probe stone, 2026-11-19). `effective`.
#define VOID_FACING 0.02f

__device__ __forceinline__ float vdepth(const Scene& S, float x, float y, float z) {
    return -((x - S.o[0]) * S.n[0] + (y - S.o[1]) * S.n[1] + (z - S.o[2]) * S.n[2]);
}

// The point's id cell (lowest corner, box index); false outside the box.
__device__ __forceinline__ bool vcell(const Void& V, float x, float y, float z, int& i, int& j, int& k) {
    i = (int)floorf(x / V.h) - V.ax; j = (int)floorf(y / V.h) - V.ay; k = (int)floorf(z / V.h) - V.az;
    return i >= 0 && j >= 0 && k >= 0 && i < V.nx - 1 && j < V.ny - 1 && k < V.nz - 1;
}

// The box's level at a point of cell (i, j, k): the earliest of the 8 corner crystals less the
// last reach; g (may be null) receives that crystal's time gradient.
__device__ float vlevel(const Void& V, float x, float y, float z, int i, int j, int k, float* g) {
    long long sy = V.nz, sx = (long long)V.ny * V.nz, b = i * sx + j * sy + k;
    float best = 3.0e38f, gc[3];
    for (int c = 0; c < 8; ++c) {
        int id = V.ids[b + ((c >> 2) & 1) * sx + ((c >> 1) & 1) * sy + (c & 1)];
        // agatesim ADR 0092: rows of AGATESIM_CRYSTAL_ROW numbers, the stage's face count
        float t = agatesim_crystal_time_grad<float>(x, y, z, V.rows + AGATESIM_CRYSTAL_ROW * id,
                                                    V.cot_a, V.elong, V.faces, g ? gc : (float*)0);
        if (t < best) { best = t; if (g) { g[0] = gc[0]; g[1] = gc[1]; g[2] = gc[2]; } }
    }
    return best - V.last;
}

// phi read trilinearly (nodes (2 a + index) h / 2), as voidbox.trilinear; NaN off its grid.
// g (may be null) receives its gradient within the cell.
__device__ float vphi(const Void& V, float x, float y, float z, float* g) {
    float hh = 0.5f * V.h;
    float qx = x / hh, qy = y / hh, qz = z / hh;
    float fx = floorf(qx), fy = floorf(qy), fz = floorf(qz);
    int i = (int)fx - 2 * V.ax, j = (int)fy - 2 * V.ay, k = (int)fz - 2 * V.az;
    if (i < 0 || j < 0 || k < 0 || i >= V.pnx - 1 || j >= V.pny - 1 || k >= V.pnz - 1)
        return __int_as_float(0x7fc00000);
    float a = qx - fx, b = qy - fy, c = qz - fz;
    long long sy = V.pnz, sx = (long long)V.pny * V.pnz;
    const float* q = V.phi + i * sx + j * sy + k;
    float c000 = q[0], c001 = q[1], c010 = q[sy], c011 = q[sy + 1];
    float c100 = q[sx], c101 = q[sx + 1], c110 = q[sx + sy], c111 = q[sx + sy + 1];
    float x00 = c000 + a * (c100 - c000), x01 = c001 + a * (c101 - c001);
    float x10 = c010 + a * (c110 - c010), x11 = c011 + a * (c111 - c011);
    float y0 = x00 + b * (x10 - x00), y1 = x01 + b * (x11 - x01);
    if (g) {
        float d00 = c100 - c000, d01 = c101 - c001, d10 = c110 - c010, d11 = c111 - c011;
        float e0 = d00 + b * (d10 - d00), e1 = d01 + b * (d11 - d01);
        g[0] = (e0 + c * (e1 - e0)) / hh;
        g[1] = ((x10 - x00) + c * ((x11 - x01) - (x10 - x00))) / hh;
        g[2] = (y1 - y0) / hh;
    }
    return y0 + c * (y1 - y0);
}

// Is the point air? v[0] receives the level, v[1] phi (+inf where none), when computed.
// (+inf for a value not reached: the gate decided first, or the point lies off the box.)
__device__ bool air_at(const Void& V, const Scene& S, float x, float y, float z, float* v) {
    v[0] = v[1] = __int_as_float(0x7f800000);
    if (V.kind == 0) return false;
    if (V.kind == 2) {
        float d = vdepth(S, x, y, z);
        v[0] = fminf(d - V.p[0], V.p[1] - d);
        return v[0] > 0.f;
    }
    if (V.kind == 3) {
        float dx = x - V.p[0], dy = y - V.p[1], dz = z - V.p[2];
        v[0] = V.p[3] - sqrtf(dx * dx + dy * dy + dz * dz);
        return v[0] > 0.f;
    }
    if (V.kind == 4) {
        float a = (x - V.p[0]) * V.p[3] + (y - V.p[1]) * V.p[4] + (z - V.p[2]) * V.p[5];
        v[0] = fminf(a, vdepth(S, x, y, z));
        return v[0] > 0.f;
    }
    int i, j, k;
    if (!vcell(V, x, y, z, i, j, k)) return false;
    long long sy = V.nz, sx = (long long)V.ny * V.nz, b = i * sx + j * sy + k;
    for (int c = 0; c < 8; ++c)
        if (!(V.nf[b + ((c >> 2) & 1) * sx + ((c >> 1) & 1) * sy + (c & 1)] & VOID_GATE)) return false;
    v[0] = vlevel(V, x, y, z, i, j, k, (float*)0);
    if (!(v[0] > 0.f)) return false;
    if (V.phi != nullptr) {
        v[1] = vphi(V, x, y, z, (float*)0);
        if (!(v[1] > 0.f)) return false;
    }
    return true;
}

// Within the shell (nearest node's flag): material is read below the slab here (kind 1 only).
__device__ __forceinline__ float vstep(const Void& V) { return V.kind == 1 ? 0.5f * V.h : V.p[7]; }

__device__ bool in_shell(const Void& V, float x, float y, float z) {
    if (V.kind != 1) return false;
    int i = (int)rintf(x / V.h) - V.ax, j = (int)rintf(y / V.h) - V.ay, k = (int)rintf(z / V.h) - V.az;
    if (i < 0 || j < 0 || k < 0 || i >= V.nx || j >= V.ny || k >= V.nz) return false;
    return V.nf[i * ((long long)V.ny * V.nz) + j * (long long)V.nz + k] & VOID_SHELL;
}

// The wall's unit normal (into the air) at a crossing, from the value that changes sign across
// the bracket: air point (xa, ya, za), solid point (xs, ys, zs) with values vs. Returns false
// when neither does (the gate alone decided: the normal is the level's, a guess).
__device__ bool wall_normal(const Void& V, const Scene& S, float xa, float ya, float za,
                            const float* vs, float* m) {
    bool sure = true;
    if (V.kind == 2) {
        float d = vdepth(S, xa, ya, za), s = (d - V.p[0] < V.p[1] - d) ? -1.f : 1.f;
        m[0] = s * S.n[0]; m[1] = s * S.n[1]; m[2] = s * S.n[2];
        return true;
    }
    if (V.kind == 3) {
        m[0] = V.p[0] - xa; m[1] = V.p[1] - ya; m[2] = V.p[2] - za;
    } else if (V.kind == 4) {
        m[0] = V.p[3]; m[1] = V.p[4]; m[2] = V.p[5];
    } else {
        bool usephi = vs[0] > 0.f && vs[1] <= 0.f;
        sure = !(vs[0] > 0.f) || usephi;
        if (usephi) vphi(V, xa, ya, za, m);
        else { int i, j, k; vcell(V, xa, ya, za, i, j, k); vlevel(V, xa, ya, za, i, j, k, m); }
    }
    float q = rsqrtf(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
    m[0] *= q; m[1] *= q; m[2] *= q;
    return sure;
}

// Reflect or refract direction d at a wall with unit normal nn facing the side d comes from
// (d . nn < 0), indices n1 (that side) and n2. Returns true when it refracts; xi chooses by the
// unpolarised Fresnel reflectance (total internal reflection reflects).
__device__ bool fresnel_turn(float* d, const float* nn0, float n1, float n2, float xi) {
    float ci = -(d[0] * nn0[0] + d[1] * nn0[1] + d[2] * nn0[2]);
    // a normal on the wrong side (a guessed one, or a grazing path) is turned to face d
    float sg = ci < 0.f ? -1.f : 1.f, nn[3] = {sg * nn0[0], sg * nn0[1], sg * nn0[2]};
    ci = fabsf(ci);
    if (xi <= fresnel(ci, n1, n2)) {
        for (int c = 0; c < 3; ++c) d[c] += 2.f * ci * nn[c];
        return false;
    }
    float eta = n1 / n2, k = 1.f - eta * eta * (1.f - ci * ci);
    float f = eta * ci - sqrtf(fmaxf(k, 0.f));
    for (int c = 0; c < 3; ++c) d[c] = eta * d[c] + f * nn[c];
    float q = rsqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    for (int c = 0; c < 3; ++c) d[c] *= q;
    return true;
}

// Bisection of a bracket along x + d s: sa air, ss solid. Leaves both ends classified as such
// (vs: the values at the solid end).
__device__ void bisect(const Void& V, const Scene& S, float x, float y, float z, float dx, float dy,
                       float dz, float& sa, float& ss, float* vs) {
    float v[2];
    for (int it = 0; it < 18; ++it) {
        float sm = 0.5f * (sa + ss);
        if (sm == sa || sm == ss) break;
        if (air_at(V, S, x + dx * sm, y + dy * sm, z + dz * sm, v)) sa = sm;
        else { ss = sm; vs[0] = v[0]; vs[1] = v[1]; }
    }
}

// Test kernels: the air rule at points, and the wall's turn (checks).
extern "C" __global__ void air_points(long long n, const float* pts, const Scene* Sp,
                                      const Void* Vp, unsigned char* out, float* vals) {
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;
    float v[2], x = pts[3 * q], y = pts[3 * q + 1], z = pts[3 * q + 2];
    out[q] = (air_at(*Vp, *Sp, x, y, z, v) ? 1 : 0) | (in_shell(*Vp, x, y, z) ? 2 : 0);
    vals[2 * q] = v[0]; vals[2 * q + 1] = v[1];
}

extern "C" __global__ void turn_points(long long n, const float* dirs, const float* normals,
                                       float n1, float n2, float xi, float* out) {
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;
    float d[3] = {dirs[3 * q], dirs[3 * q + 1], dirs[3 * q + 2]};
    out[4 * q + 3] = fresnel_turn(d, normals + 3 * q, n1, n2, xi) ? 1.f : 0.f;
    out[4 * q] = d[0]; out[4 * q + 1] = d[1]; out[4 * q + 2] = d[2];
}
