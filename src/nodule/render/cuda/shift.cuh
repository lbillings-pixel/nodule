// Materials that vary along the bands (ADR 0016; agatesim ADR 0078): a GPU copy of agatesim
// volume.features MaterialVariation.shift, worked out at each sample from its restored place
// and growth time. Field 0 is the pigment lenses, field 1 the translucency drift. Each is a
// SheetNoise: growth depth runs through uneven sheets, each with its own pattern, blended
// keeping the variance; a pattern is one octave of agatesim BandNoise, a periodic cubic
// B-spline tile read through a rotation and offset (scipy map_coordinates, order 3,
// grid-wrap, prefilter off). The host folds each sheet's offset into one constant per sheet
// (shift.py), so the index is M p + c_j with M = rotation / cell. tests/test_render_gpu.py
// checks this against agatesim's own material_shift, so a change there shows up.

struct Shift {
    const float* depth;      // growth depth (mm) at the steps + 1 step edges
    const float* sheets[2];  // per field: sheet depths (mm)
    const float* cj[2];      // per field: nsheet x 3 index constants
    const float* coef[2];    // per field: n^3 spline coefficients, C order
    const float* band;       // per material: translucency (-1: takes no shift), pigment mua x 3
    const float* dgrid;      // per cut: the drift's change baked on the scene's growth-time
                             // grid (ADR 0016 s.9); null: worked out at each sample
    float M[2][9];           // per field: rotation / cell, rows
    int nsheet[2], n[2], on[2];
    int steps, tr0, ntr;     // step count; translucency table rows (ntr 0: no drift table)
    float strength, threshold, drift;
};

#define LENS_EDGE 0.5f    // agatesim features.LENS_EDGE
#define DRIFT_RANGE 2.0f  // agatesim features.DRIFT_RANGE

__device__ __forceinline__ void bspline3(float t, float* w) {   // cubic B-spline weights
    float s = 1.f - t, t2 = t * t, t3 = t2 * t;
    w[0] = s * s * s * (1.f / 6.f);
    w[1] = (3.f * t3 - 6.f * t2 + 4.f) * (1.f / 6.f);
    w[2] = (-3.f * t3 + 3.f * t2 + 3.f * t + 1.f) * (1.f / 6.f);
    w[3] = t3 * (1.f / 6.f);
}

// The periodic spline at fractional index (a, b, c); taps floor - 1 .. floor + 2, wrapped.
// n is a power of two (shift.py checks), so wrapping is a mask, also for negative indices.
__device__ float tile_at(const float* __restrict__ coef, int n, float a, float b, float c) {
    float fa = floorf(a), fb = floorf(b), fc = floorf(c);
    float wa[4], wb[4], wc[4];
    bspline3(a - fa, wa); bspline3(b - fb, wb); bspline3(c - fc, wc);
    int ia = (int)fa - 1, ib = (int)fb - 1, ic = (int)fc - 1, mask = n - 1;
    int ka[4], kb[4], kc[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) { ka[q] = (ia + q) & mask; kb[q] = (ib + q) & mask; kc[q] = (ic + q) & mask; }
    float out = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float* pa = coef + ka[i] * n * n;
        float si = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float* pb = pa + kb[j] * n;
            si += wb[j] * (wc[0] * __ldg(pb + kc[0]) + wc[1] * __ldg(pb + kc[1])
                           + wc[2] * __ldg(pb + kc[2]) + wc[3] * __ldg(pb + kc[3]));
        }
        out += wa[i] * si;
    }
    return out;
}

__device__ __forceinline__ float pattern(const Shift& H, int f, int j, float x, float y, float z) {
    const float* M = H.M[f]; const float* c = H.cj[f] + 3 * j;
    return tile_at(H.coef[f], H.n[f], M[0] * x + M[1] * y + M[2] * z + c[0],
                   M[3] * x + M[4] * y + M[5] * z + c[1], M[6] * x + M[7] * y + M[8] * z + c[2]);
}

// agatesim SheetNoise.__call__ at one point and growth depth.
__device__ float sheet_noise(const Shift& H, int f, float x, float y, float z, float depth) {
    const float* s = H.sheets[f];
    int lo = 0, hi = H.nsheet[f];                       // count of sheets <= depth
    while (lo < hi) { int mid = (lo + hi) >> 1; if (s[mid] <= depth) lo = mid + 1; else hi = mid; }
    int j = min(max(lo - 1, 0), H.nsheet[f] - 2);
    float u = fminf(fmaxf((depth - s[j]) / (s[j + 1] - s[j]), 0.f), 1.f);
    float angle = 1.57079633f * u * u * (3.f - 2.f * u);
    return cosf(angle) * pattern(H, f, j, x, y, z) + sinf(angle) * pattern(H, f, j + 1, x, y, z);
}

// MaterialVariation.shift: the pigment multiple and the translucency change at a restored point
// with growth time t (Fields.t_smooth scale). lens / drift false skip a field (gain 1, change 0).
__device__ void shift_at(const Shift& H, float x, float y, float z, float t, bool lens, bool drift,
                         float& gain, float& change) {
    gain = 1.f; change = 0.f;
    float k = fminf(fmaxf(t * H.steps, 0.f), (float)H.steps);   // np.interp clamps at the ends
    int i = min((int)k, H.steps - 1);
    float depth = H.depth[i] + (k - i) * (H.depth[i + 1] - H.depth[i]);
    if (lens && H.on[0]) {
        float u = (sheet_noise(H, 0, x, y, z, depth) - H.threshold) / LENS_EDGE + 0.5f;
        u = fminf(fmaxf(u, 0.f), 1.f);
        gain += H.strength * u * u * (3.f - 2.f * u);
    }
    if (drift && H.on[1])
        change = H.drift * fminf(fmaxf(sheet_noise(H, 1, x, y, z, depth) / DRIFT_RANGE, -1.f), 1.f);
}

extern "C" __global__ void shift_points(long long n, const float* pts, const float* ts,
                                        const Shift* Hp, float* gain, float* change) {
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    float g, c;
    shift_at(*Hp, pts[3 * p], pts[3 * p + 1], pts[3 * p + 2], ts[p], true, true, g, c);
    gain[p] = g; change[p] = c;
}
