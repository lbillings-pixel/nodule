// Forward Monte Carlo through a slab 0 < z < d (d = inf: half-space), normal beam at the origin.
// The independent reference for the camera kernel (reciprocity check) and the MCML checks
// (Wang 1995 Tables 1-2). One thread per packet; null-collision tracking against mubar with
// Kutz spectral tracking; smooth dielectric faces, all-or-none Fresnel; n_out above the face,
// n_bot below the back face (n_bot = n_in: an index-matched, black bottom). Media: one material, or
// a material-id grid along z (nearest cell; for the band phantom). Needs common.cuh first.

__device__ int zmaterial(const unsigned char* grid, int gz, float h, float z) {
    if (gz == 0) return 0;
    int k = min(max((int)floorf(z / h), 0), gz - 1);
    return grid[k];
}

// analog = 1: unweighted delta tracking (Woodcock 1965), biased when mubar < mu_t (control).
extern "C" __global__ void slab(
    long long n, unsigned int seed, float d, float n_in, float n_out, float n_bot, float g,
    const float* mua, const float* mus, float mubar, int analog,
    const unsigned char* grid, int gz, float h, float* out /* per packet: Rd[3], Tt[3], events */)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    Rng r; r.init(seed, p);
    float rsp = fresnel(1.f, n_out, n_in);
    float w[3] = {1.f - rsp, 1.f - rsp, 1.f - rsp};
    float z = 0.f, ux = 0.f, uy = 0.f, uz = 1.f;
    float R[3] = {0.f, 0.f, 0.f}, T[3] = {0.f, 0.f, 0.f};
    int events = 0;
    for (;;) {
        float s = -logf(r.next()) / mubar;
        float zb = uz > 0.f ? d : (uz < 0.f ? 0.f : __int_as_float(0x7f800000));
        float db = uz != 0.f ? (zb - z) / uz : __int_as_float(0x7f800000);
        if (s >= db) {                                   // reach a face
            z = zb;
            if (r.next() <= fresnel(fabsf(uz), n_in, uz < 0.f ? n_out : n_bot)) { uz = -uz; continue; }
            float* dst = uz < 0.f ? R : T;
            for (int c = 0; c < 3; ++c) dst[c] += w[c];
            break;
        }
        z += uz * s; ++events;
        int m = zmaterial(grid, gz, h, z);
        if (analog) {
            float ms = mus[3 * m], mt = mua[3 * m] + ms;
            if (r.next() < mt / mubar) {
                for (int c = 0; c < 3; ++c) w[c] *= mus[3 * m + c] / (mua[3 * m + c] + mus[3 * m + c]);
                hg(g, ux, uy, uz, r);
            }
        } else if (spectral_step(w, mua + 3 * m, mus, m, mubar, r)) {
            hg(g, ux, uy, uz, r);
        }
        if (!roulette(w, 1.f, r)) break;
        if (events > 100000) break;
    }
    float* o = out + 7 * p;
    for (int c = 0; c < 3; ++c) { o[c] = R[c]; o[3 + c] = T[c]; }
    o[6] = (float)events;
}
