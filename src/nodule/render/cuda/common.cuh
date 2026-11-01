// Shared device code for nodule.render (ADR 0007): random numbers, Fresnel, phase function.
// Units: mm, per mm. Compiled by NVRTC through CuPy; the Python side joins the sources.

// Philox4x32-10 (Salmon et al. 2011). A packet's stream is keyed by (seed, packet id) and the
// counter steps once per four draws, so a packet draws the same numbers whatever the launch size.
struct u4 { unsigned int x, y, z, w; };
__device__ __forceinline__ u4 philox(u4 c, unsigned int k0, unsigned int k1) {
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
        unsigned int hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
        unsigned int hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
        u4 n; n.x = hi1 ^ c.y ^ k0; n.y = lo1; n.z = hi0 ^ c.w ^ k1; n.w = lo0;
        c = n; k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    return c;
}
struct Rng {
    unsigned int k0, k1, hi, ctr; u4 buf; int left;
    __device__ void init(unsigned int seed, long long id) {
        k0 = seed; k1 = (unsigned int)id; hi = (unsigned int)(id >> 32); ctr = 0; left = 0;
    }
    __device__ float next() {
        if (left == 0) { u4 c; c.x = ctr++; c.y = hi; c.z = 0; c.w = 0;
                         buf = philox(c, k0, k1); left = 4; }
        unsigned int v = (left == 4) ? buf.x : (left == 3) ? buf.y : (left == 2) ? buf.z : buf.w;
        --left;
        return ((v >> 9) + 0.5f) * (1.0f / 8388608.0f);   // open (0, 1): never 0 or 1 in float32
    }
};

// Unpolarised Fresnel reflectance (Wang 1995 Eq. 3.30); ci = |cos| of incidence; TIR gives 1.
__device__ float fresnel(float ci, float n1, float n2) {
    float eta = n1 / n2, s2 = eta * eta * (1.f - ci * ci);
    if (s2 >= 1.f) return 1.f;
    float ct = sqrtf(1.f - s2);
    float rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct);
    float rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci);
    return 0.5f * (rs * rs + rp * rp);
}

// Henyey-Greenstein sampling (Wang 1995 Eq. 3.22) and density per steradian.
// The cosine is formed as 1 - (1 - ct): with s = 1 - g + 2 g xi and t = (1 - g^2)/s,
// 1 - ct = (1 - g)(1 - xi)(t + 1 - g)/s exactly, which keeps float32 accurate up to g ~ 0.999
// (the textbook (1 + g^2 - t^2)/2g cancels badly there; Apetz grains reach g 0.998).
__device__ void hg(float g, float& ux, float& uy, float& uz, Rng& r) {
    float ct, omc;
    float xi = r.next();
    if (fabsf(g) < 1e-6f) { ct = 2.f * xi - 1.f; omc = 2.f - 2.f * xi; }
    else { float s = 1.f - g + 2.f * g * xi, t = (1.f - g * g) / s;
           omc = (1.f - g) * (1.f - xi) * (t + 1.f - g) / s;
           omc = fminf(2.f, fmaxf(0.f, omc)); ct = 1.f - omc; }
    float st = sqrtf(omc * (2.f - omc)), ph = 6.28318531f * r.next();
    float cp = cosf(ph), sp = sinf(ph);
    // copysignf here once lost the sign at normal incidence (Rd 25% low): keep the explicit test
    if (fabsf(uz) > 0.99999f) { ux = st * cp; uy = st * sp; uz = uz > 0.f ? ct : -ct; return; }
    float den = sqrtf(1.f - uz * uz);
    float nx = st * (ux * uz * cp - uy * sp) / den + ux * ct;
    float ny = st * (uy * uz * cp + ux * sp) / den + uy * ct;
    float nz = -st * cp * den + uz * ct;
    // renormalise: float32 drift over hundreds of scatters otherwise ends in NaN
    float q = rsqrtf(nx * nx + ny * ny + nz * nz);
    ux = nx * q; uy = ny * q; uz = nz * q;
}
__device__ __forceinline__ float hgpdf(float g, float c) {
    float d = (1.f - g) * (1.f - g) + 2.f * g * (1.f - c);   // = 1 + g^2 - 2gc, without cancellation
    return (1.f - g * g) / (12.5663706f * d * sqrtf(d));
}

// Two-lobe phase function per material and channel (ADR 0007 7a C): ph[9m + 3c + {0,1,2}] =
// {g_a, f, g_b}, p_c(cos) = (1 - f) HG(g_a) + f HG(g_b). Lobe a is the grain part, lobe b the
// pores and other near-isotropic scatterers.
__device__ __forceinline__ float phase_pdf(const float* ph, int m, int c, float cs) {
    const float* q = ph + 9 * m + 3 * c;
    return (1.f - q[1]) * hgpdf(q[0], cs) + q[1] * hgpdf(q[2], cs);
}
__device__ __forceinline__ bool same_lobes(const float* ph, int m) {   // all channels alike
    const float* q0 = ph + 9 * m;
    bool same = true;
    for (int k = 3; k < 9; ++k) same = same && q0[k] == q0[k % 3];
    return same;
}
// Density per steradian with which `scatter` (below) draws a new direction at cosine cs from
// the old one, given the weights w before the draw. Used for multiple importance sampling.
__device__ float sample_pdf(const float* ph, int m, const float* w, float cs) {
    if (same_lobes(ph, m)) return phase_pdf(ph, m, 0, cs);
    float a0 = fabsf(w[0]), a1 = fabsf(w[1]), a2 = fabsf(w[2]), sa = a0 + a1 + a2;
    if (!(sa > 0.f)) { a0 = a1 = a2 = 1.f; sa = 3.f; }
    return (a0 * phase_pdf(ph, m, 0, cs) + a1 * phase_pdf(ph, m, 1, cs)
            + a2 * phase_pdf(ph, m, 2, cs)) / sa;
}
// Scatter at a real collision: sample one channel's phase function (channel chosen by |w|),
// then reweight every channel by p_c / p_mix, p_mix = sum_c pi_c p_c (the one-sample mixture
// estimator; unbiased, and the weights stay bounded by 1/pi_c). When all channels share one
// lobe the ratio is 1 and this is plain HG sampling. If pdf is given it receives the density
// of the drawn direction (sample_pdf).
__device__ void scatter(const float* ph, int m, float* w, float& ux, float& uy, float& uz, Rng& r,
                        bool reweight = true, float* pdf = nullptr) {
    const float* q0 = ph + 9 * m;
    if (same_lobes(ph, m)) {                               // weight ratio 1; a single lobe draws
        float g = q0[0];                                   // exactly as plain HG sampling did
        if (q0[1] > 0.f && r.next() < q0[1]) g = q0[2];
        float ox = ux, oy = uy, oz = uz;
        hg(g, ux, uy, uz, r);
        if (pdf) *pdf = phase_pdf(ph, m, 0, ox * ux + oy * uy + oz * uz);
        return;
    }
    float a0 = fabsf(w[0]), a1 = fabsf(w[1]), a2 = fabsf(w[2]), sa = a0 + a1 + a2;
    if (!(sa > 0.f)) { a0 = a1 = a2 = 1.f; sa = 3.f; }
    float u = r.next() * sa;
    int c = u < a0 ? 0 : (u < a0 + a1 ? 1 : 2);
    const float* q = q0 + 3 * c;
    float g = r.next() < q[1] ? q[2] : q[0];
    float ox = ux, oy = uy, oz = uz;
    hg(g, ux, uy, uz, r);
    float cs = ox * ux + oy * uy + oz * uz;
    float p[3], pm = 0.f, pi[3] = {a0 / sa, a1 / sa, a2 / sa};
    for (int k = 0; k < 3; ++k) { p[k] = phase_pdf(ph, m, k, cs); pm += pi[k] * p[k]; }
    if (pdf) *pdf = pm;
    if (reweight) for (int k = 0; k < 3; ++k) w[k] *= p[k] / pm;   // off: a check's control
}

// Kutz et al. 2017 spectral tracking at a tentative collision, three channels sharing one path.
// History-aware probability of a real collision; weights updated in place. Returns 1 if real.
__device__ int spectral_step(float* w, const float* mua, const float* mus, int m, float mubar,
                             Rng& r) {
    float mt[3], mn[3], ms[3], sr = 0.f, sn = 0.f;
    for (int c = 0; c < 3; ++c) {
        ms[c] = mus[3 * m + c]; mt[c] = mua[3 * m + c] + ms[c]; mn[c] = mubar - mt[c];
        sr += fabsf(w[c] * mt[c]); sn += fabsf(w[c] * mn[c]);
    }
    float pr = sr / (sr + sn);
    if (r.next() < pr) {                       // real: always scatter, implicit capture
        for (int c = 0; c < 3; ++c) w[c] *= ms[c] / (mubar * pr);
        return 1;
    }
    for (int c = 0; c < 3; ++c) w[c] *= mn[c] / (mubar * (1.f - pr));
    return 0;
}

// Russian roulette, W_th = 1e-4 of the start weight, survival 1/10 (Wang 1995 Eq. 3.36).
// Returns 0 when the packet ends (also for zero weight, which must not survive).
__device__ int roulette(float* w, float w0, Rng& r) {
    float wm = fmaxf(fabsf(w[0]), fmaxf(fabsf(w[1]), fabsf(w[2])));
    if (wm == 0.f) return 0;
    if (wm < 1e-4f * w0) {
        if (r.next() < 0.1f) { for (int c = 0; c < 3; ++c) w[c] *= 10.f; }
        else return 0;
    }
    return 1;
}
