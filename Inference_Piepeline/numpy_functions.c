// numpy_compat_f32.c
// Reference-accurate (single-precision) C equivalents of:
//   - np.fft.fft (1D forward, crop/zero-pad, norm={"backward","forward","ortho"})
//   - np.fft.rfftfreq
//   - scipy.signal.find_peaks (core behavior incl. plateau/height/threshold/distance/prominence/width)
//   - scipy.signal.medfilt (N-D, zero padding, odd kernel)
// All math uses 32-bit floats; ints are 32-bit. Suitable for 32-bit CPUs.
//
// Build (GCC/Clang):
//   gcc -std=c11 -O2 -c numpy_compat_f32.c -o numpy_compat_f32.o
//   gcc -std=c11 -O2 -DNUMPY_COMPAT_MAIN numpy_compat_f32.c -o demo -lm
//
// Notes:
// - FFT is a direct DFT (O(n^2)) to exactly match semantics for any n.
// - Prominence/width logic matches SciPy behavior closely; includes width fallbacks to bases.
// - medfilt_nd is correct but not optimized (reference implementation).

#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <complex.h>
#include <stdint.h>
#include "numpy_functions.h"

#ifndef M_PIf
#define M_PIf 3.14159274101257324219f
#endif

static inline float _complex_real(float _Complex z){ return crealf(z); }
static inline float _complex_imag(float _Complex z){ return cimagf(z); }
// Public API declarations are provided in `numpy_functions.h`.
// Implementations follow below.

// ============================ Utilities (internal) ============================

static inline int32_t imin32(int32_t a, int32_t b){ return a < b ? a : b; }
static inline int32_t imax32(int32_t a, int32_t b){ return a > b ? a : b; }
static inline float  fmin32(float a, float b){ return a < b ? a : b; }
static inline float  fmax32(float a, float b){ return a > b ? a : b; }

static void* xmalloc(uint32_t n){
    void* p = malloc((size_t)n);
    if(!p){ fprintf(stderr, "Out of memory\n"); exit(1); }
    return p;
}

static void* xmalloc_bytes(size_t n){
    void* p = malloc(n);
    if(!p){ fprintf(stderr, "Out of memory\n"); exit(1); }
    return p;
}

static void* xrealloc(void* ptr, size_t n){
    void* q = realloc(ptr, n);
    if(!q){ free(ptr); fprintf(stderr, "Out of memory (realloc)\n"); exit(1); }
    return q;
}

// ============================ FFT (DFT-based) ============================

fft_norm_t parse_fft_norm(const char* norm){
    if(!norm || strcmp(norm, "backward")==0) return FFT_NORM_BACKWARD;
    if(strcmp(norm, "forward")==0)  return FFT_NORM_FORWARD;
    if(strcmp(norm, "ortho")==0)    return FFT_NORM_ORTHO;
    return FFT_NORM_BACKWARD;
}

// Complex multiply in float without calling library:
static inline float _Complex cmulf32(float _Complex a, float _Complex b){
    float ar = _complex_real(a), ai = _complex_imag(a);
    float br = _complex_real(b), bi = _complex_imag(b);
    return (ar*br - ai*bi) + (ar*bi + ai*br)*I;
}

// FFT implementation intentionally removed. Other functions (rfftfreq, find_peaks_ref,
// medfilt_nd) remain in this file. If you need an FFT implementation, consider
// adding PocketFFT in `vendor/pocketfft/` and wiring it here for NumPy-compatible
// results.

// ============================ rfftfreq ============================

float* np_fft_rfftfreq(int32_t n, float d){
    if(n <= 0) return NULL;
    // Determine length of frequency array
    int32_t N = n/2 + 1;  // works for both even and odd n
    float* f = (float*)xmalloc((uint32_t)(sizeof(float) * (uint32_t)N));
    // Compute frequency step in float precision to match NumPy behavior, then cast to float
    float step_d = 1.0 / ((float)n * (float)d);
    for(int32_t i = 0; i < N; ++i) {
        float val = (float)i * step_d;
        f[i] = (float)val;
    }
    return f;
}

// ============================ find_peaks ============================

typedef struct { bool has_min, has_max; float vmin, vmax; } interval_t;

// after (renamed parameter to iv)
static inline bool in_interval_f(float x, interval_t iv){
    if(iv.has_min && x < iv.vmin) return false;
    if(iv.has_max && x > iv.vmax) return false;
    return true;
}

// Local maxima with plateau handling.
// Output arrays (malloc'd): peaks, left_edges, right_edges; count in out_n.
static void local_maxima_with_plateaus_f(const float* x, int32_t n,
                                         int32_t** out_peaks, int32_t* out_n,
                                         int32_t** out_left_edges,
                                         int32_t** out_right_edges){
    int32_t cap = imax32(8, n/4 + 1);
    int32_t* peaks = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)cap));
    int32_t* L     = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)cap));
    int32_t* R     = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)cap));
    int32_t k = 0;

    int32_t i = 1;
    while(i < n-1){
        if(x[i] > x[i-1] && x[i] > x[i+1]){
            if(k == cap){
                cap *= 2;
                peaks = (int32_t*)xrealloc(peaks, (size_t)sizeof(int32_t)*(size_t)cap);
                L     = (int32_t*)xrealloc(L,     (size_t)sizeof(int32_t)*(size_t)cap);
                R     = (int32_t*)xrealloc(R,     (size_t)sizeof(int32_t)*(size_t)cap);
            }
            peaks[k]=i; L[k]=i; R[k]=i; k++; i++; continue;
        }
        if(x[i] >= x[i-1] && x[i] == x[i+1]){
            int32_t s = i;
            int32_t e = i+1;
            while(e < n-1 && x[e] == x[e+1]) e++;
            float left  = x[s-1];
            float right = x[e+1];
            float level = x[i];
            if(level > left && level > right){
                int32_t mid = (s + e) / 2; // round down
                if(k == cap){
                    cap *= 2;
                    peaks = (int32_t*)xrealloc(peaks, (size_t)sizeof(int32_t)*(size_t)cap);
                    L     = (int32_t*)xrealloc(L,     (size_t)sizeof(int32_t)*(size_t)cap);
                    R     = (int32_t*)xrealloc(R,     (size_t)sizeof(int32_t)*(size_t)cap);
                }
                peaks[k]=mid; L[k]=s; R[k]=e; k++;
            }
            i = e+1; continue;
        }
        i++;
    }

    *out_peaks = peaks; *out_n = k; *out_left_edges = L; *out_right_edges = R;
}

// Neighbor thresholds (x[p]-x[p-1], x[p]-x[p+1])
static void compute_thresholds_f(const float* x, const int32_t* peaks, int32_t n_peaks,
                                 float* left_thr, float* right_thr){
    for(int32_t i=0;i<n_peaks;++i){
        int32_t p = peaks[i];
        left_thr[i]  = x[p] - x[p-1];
        right_thr[i] = x[p] - x[p+1];
    }
}

// Enforce minimal distance: keep higher peaks first.
static int32_t* select_by_distance_f(const int32_t* peaks, int32_t n_peaks,
                                     const float* x, int32_t distance, int32_t* out_n){
    int32_t* order = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)n_peaks));
    for(int32_t i=0;i<n_peaks;++i) order[i]=i;

    // selection sort by descending height
    for(int32_t i=0;i<n_peaks;++i){
        int32_t best=i;
        for(int32_t j=i+1;j<n_peaks;++j){
            if(x[peaks[order[j]]] > x[peaks[order[best]]]) best=j;
        }
        int32_t tmp=order[i]; order[i]=order[best]; order[best]=tmp;
    }

    bool* taken = (bool*)xmalloc((uint32_t)(sizeof(bool)*(uint32_t)n_peaks));
    memset(taken, 0, (size_t)n_peaks*sizeof(bool));

    int32_t* keep = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)n_peaks));
    int32_t k=0;
    for(int32_t oi=0; oi<n_peaks; ++oi){
        int32_t idx = order[oi];
        if(taken[idx]) continue;
        int32_t p = peaks[idx];
        keep[k++] = idx;
        for(int32_t j=0;j<n_peaks;++j){
            if(taken[j] || j==idx) continue;
            if(abs(peaks[j]-p) < distance) taken[j] = true;
        }
    }
    free(order); free(taken);

    // compress to ascending peak positions
    int32_t* sel = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)k));
    for(int32_t i=0;i<k;++i) sel[i] = peaks[ keep[i] ];
    for(int32_t i=0;i<k;++i)
        for(int32_t j=i+1;j<k;++j)
            if(sel[j] < sel[i]){ int32_t t=sel[i]; sel[i]=sel[j]; sel[j]=t; }

    *out_n = k;
    free(keep);
    return sel;
}

// Prominence within optional window wlen
static void compute_prominence_f(const float* x, int32_t n,
                                 const int32_t* peaks, int32_t n_peaks,
                                 int32_t wlen,
                                 float* prominences, int32_t* left_bases, int32_t* right_bases){
    for(int32_t i=0;i<n_peaks;++i){
        int32_t p = peaks[i];
        float peak_h = x[p];
        int32_t left_limit = 0, right_limit = n-1;
        if(wlen > 0){
            int32_t half = wlen/2;
            left_limit  = imax32(0, p - half);
            right_limit = imin32(n-1, p + half);
        }
        float min_left = peak_h; int32_t lb = left_limit;
        float cur_min = peak_h;
        for(int32_t j=p; j>=left_limit; --j){
            if(x[j] > peak_h) break;
            if(x[j] < cur_min){ cur_min = x[j]; lb = j; }
        }
        min_left = cur_min;

        float min_right = peak_h; int32_t rb = right_limit;
        cur_min = peak_h;
        for(int32_t j=p; j<=right_limit; ++j){
            if(x[j] > peak_h) break;
            if(x[j] < cur_min){ cur_min = x[j]; rb = j; }
        }
        min_right = cur_min;

        float ref = fmax32(min_left, min_right);
        prominences[i] = peak_h - ref;
        left_bases[i] = lb; right_bases[i] = rb;
    }
}

// Widths at height = peak - prominence*rel_height
static void compute_widths_f(const float* x, const int32_t* peaks, int32_t n_peaks,
                             const float* prominences, const int32_t* left_bases,
                             const int32_t* right_bases, float rel_height,
                             float* widths, float* width_heights,
                             float* left_ips, float* right_ips){
    for(int32_t i=0;i<n_peaks;++i){
        int32_t p = peaks[i];
        float h = x[p] - prominences[i]*rel_height;
        width_heights[i] = h;

        // Left intersection
        float li = (float)p;
        bool found_left = false;
        for(int32_t j=p; j>left_bases[i]; --j){
            if(x[j-1] <= h && x[j] >= h){
                float y1 = x[j-1], y2 = x[j];
                float t = (h - y1) / (y2 - y1 + 1e-30f);
                li = (float)(j-1) + t;
                found_left = true;
                break;
            }
        }
        if(!found_left) li = (float)left_bases[i]; // fallback to base

        // Right intersection
        float ri = (float)p;
        bool found_right = false;
        for(int32_t j=p; j<right_bases[i]; ++j){
            if(x[j+1] <= h && x[j] >= h){
                float y1 = x[j], y2 = x[j+1];
                float t = (h - y1) / (y2 - y1 + 1e-30f);
                ri = (float)j + t;
                found_right = true;
                break;
            }
        }
        if(!found_right) ri = (float)right_bases[i]; // fallback to base

        left_ips[i] = li; right_ips[i] = ri; widths[i] = ri - li;
    }
}

peak_result_t find_peaks_ref(const float* x, int32_t n,
                             const float height[2],
                             const float threshold[2],
                             int32_t distance,
                             const float prominence[2],
                             const float width[2],
                             int32_t wlen,
                             float rel_height,
                             const float plateau_size[2]){
    peak_result_t res; memset(&res, 0, sizeof(res));
    if(n < 3) return res;
    if(distance < 0){ fprintf(stderr, "distance must be >= 0\n"); return res; }

    interval_t Iheight={0}, Ithr={0}, Iprom={0}, Iwidth={0}, Iplat={0};
    if(height){
        if(!isnan(height[0])){ Iheight.has_min=true; Iheight.vmin=height[0]; }
        if(!isnan(height[1])){ Iheight.has_max=true; Iheight.vmax=height[1]; }
    }
    if(threshold){
        if(!isnan(threshold[0])){ Ithr.has_min=true; Ithr.vmin=threshold[0]; }
        if(!isnan(threshold[1])){ Ithr.has_max=true; Ithr.vmax=threshold[1]; }
    }
    if(prominence){
        if(!isnan(prominence[0])){ Iprom.has_min=true; Iprom.vmin=prominence[0]; }
        if(!isnan(prominence[1])){ Iprom.has_max=true; Iprom.vmax=prominence[1]; }
    }
    if(width){
        if(!isnan(width[0])){ Iwidth.has_min=true; Iwidth.vmin=width[0]; }
        if(!isnan(width[1])){ Iwidth.has_max=true; Iwidth.vmax=width[1]; }
    }
    if(plateau_size){
        if(!isnan(plateau_size[0])){ Iplat.has_min=true; Iplat.vmin=plateau_size[0]; }
        if(!isnan(plateau_size[1])){ Iplat.has_max=true; Iplat.vmax=plateau_size[1]; }
    }

    // Local maxima + plateaus
    int32_t *peaks=NULL, *L=NULL, *R=NULL, n_peaks=0;
    local_maxima_with_plateaus_f(x, n, &peaks, &n_peaks, &L, &R);
    if(n_peaks==0){ free(peaks); free(L); free(R); return res; }

    // plateau_size filtering
    int32_t *keep_idx = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)n_peaks));
    int32_t keep_n=0;
    int32_t* plateau_sizes = NULL; int32_t* left_edges=NULL; int32_t* right_edges=NULL;

    if(Iplat.has_min || Iplat.has_max){
        plateau_sizes = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)n_peaks));
        for(int32_t i=0;i<n_peaks;++i) plateau_sizes[i] = R[i]-L[i]+1;
        for(int32_t i=0;i<n_peaks;++i)
            if(in_interval_f((float)plateau_sizes[i], Iplat)) keep_idx[keep_n++]=i;
    } else {
        for(int32_t i=0;i<n_peaks;++i) keep_idx[keep_n++]=i;
    }

    int32_t K=keep_n;
    int32_t* P  = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)K));
    int32_t* LL = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)K));
    int32_t* RR = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)K));
    for(int32_t i=0;i<K;++i){ P[i]=peaks[keep_idx[i]]; LL[i]=L[keep_idx[i]]; RR[i]=R[keep_idx[i]]; }
    free(peaks); free(L); free(R);

    // height
    float* peak_heights = NULL;
    if(Iheight.has_min || Iheight.has_max){
        peak_heights = (float*)xmalloc((uint32_t)(sizeof(float)*(uint32_t)K));
        int32_t w=0;
        for(int32_t i=0;i<K;++i){
            peak_heights[i]=x[P[i]];
            if(in_interval_f(peak_heights[i], Iheight)){
                P[w]=P[i]; LL[w]=LL[i]; RR[w]=RR[i];
                peak_heights[w]=peak_heights[i]; w++;
            }
        }
        K=w;
    }

    // threshold
    float *left_thr=NULL, *right_thr=NULL;
    if(Ithr.has_min || Ithr.has_max){
        left_thr  = (float*)xmalloc((uint32_t)(sizeof(float)*(uint32_t)K));
        right_thr = (float*)xmalloc((uint32_t)(sizeof(float)*(uint32_t)K));
        compute_thresholds_f(x, P, K, left_thr, right_thr);
        int32_t w=0;
        for(int32_t i=0;i<K;++i){
            float lt=left_thr[i], rt=right_thr[i];
            if(in_interval_f(lt, Ithr) && in_interval_f(rt, Ithr)){
                P[w]=P[i]; LL[w]=LL[i]; RR[w]=RR[i];
                if(peak_heights) peak_heights[w]=x[P[w]];
                left_thr[w]=lt; right_thr[w]=rt; w++;
            }
        }
        K=w;
    }

    // distance
    if(distance >= 1 && K>1){
        int32_t newN=0; int32_t* sel = select_by_distance_f(P, K, x, distance, &newN);

        int32_t* newLL=(int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)newN));
        int32_t* newRR=(int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)newN));
        float* newPH = peak_heights? (float*)xmalloc((uint32_t)(sizeof(float)*(uint32_t)newN)) : NULL;
        float* newLT = left_thr? (float*)xmalloc((uint32_t)(sizeof(float)*(uint32_t)newN)) : NULL;
        float* newRT = right_thr? (float*)xmalloc((uint32_t)(sizeof(float)*(uint32_t)newN)) : NULL;

        for(int32_t i=0;i<newN;++i){
            int32_t pos = sel[i];
            int32_t j=0; for(; j<K; ++j) if(P[j]==pos) break;
            newLL[i]=LL[j]; newRR[i]=RR[j];
            if(newPH) newPH[i]=x[pos];
            if(newLT) newLT[i]=left_thr[j];
            if(newRT) newRT[i]=right_thr[j];
        }
        free(P); free(LL); free(RR);
        if(peak_heights){ free(peak_heights); peak_heights=newPH; }
        if(left_thr){ free(left_thr); left_thr=newLT; }
        if(right_thr){ free(right_thr); right_thr=newRT; }
        P = sel; K = newN; LL=newLL; RR=newRR;
    }

    // prominence (and width needs prominence)
    float *prom=NULL; int32_t *lb=NULL, *rb=NULL;
    if(Iprom.has_min || Iprom.has_max || Iwidth.has_min || Iwidth.has_max){
        prom = (float*)xmalloc((uint32_t)(sizeof(float)*(uint32_t)K));
        lb   = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)K));
        rb   = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)K));
        compute_prominence_f(x, n, P, K, wlen, prom, lb, rb);

        if(Iprom.has_min || Iprom.has_max){
            int32_t w=0;
            for(int32_t i=0;i<K;++i){
                if(in_interval_f(prom[i], Iprom)){
                    P[w]=P[i]; LL[w]=LL[i]; RR[w]=RR[i];
                    prom[w]=prom[i]; lb[w]=lb[i]; rb[w]=rb[i];
                    if(peak_heights) peak_heights[w]=x[P[w]];
                    if(left_thr){ left_thr[w]=left_thr[i]; right_thr[w]=right_thr[i]; }
                    w++;
                }
            }
            K=w;
        }
    }

    // width
    float *widths=NULL, *wh=NULL, *lips=NULL, *rips=NULL;
    if(Iwidth.has_min || Iwidth.has_max){
        widths=(float*)xmalloc((uint32_t)(sizeof(float)*(uint32_t)K));
        wh    =(float*)xmalloc((uint32_t)(sizeof(float)*(uint32_t)K));
        lips  =(float*)xmalloc((uint32_t)(sizeof(float)*(uint32_t)K));
        rips  =(float*)xmalloc((uint32_t)(sizeof(float)*(uint32_t)K));
        compute_widths_f(x, P, K, prom, lb, rb, rel_height, widths, wh, lips, rips);

        int32_t w=0;
        for(int32_t i=0;i<K;++i){
            if(in_interval_f(widths[i], Iwidth)){
                P[w]=P[i]; LL[w]=LL[i]; RR[w]=RR[i];
                widths[w]=widths[i]; wh[w]=wh[i]; lips[w]=lips[i]; rips[w]=rips[i];
                if(peak_heights) peak_heights[w]=x[P[w]];
                if(left_thr){ left_thr[w]=left_thr[i]; right_thr[w]=right_thr[i]; }
                if(prom){ prom[w]=prom[i]; lb[w]=lb[i]; rb[w]=rb[i]; }
                w++;
            }
        }
        K=w;
    }

    // Finalize
    res.n_peaks = K;
    res.indices = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)K));
    for(int32_t i=0;i<K;++i) res.indices[i]=P[i];

    res.peak_heights = peak_heights;
    res.left_thresholds = left_thr; res.right_thresholds = right_thr;
    res.prominences = prom; res.left_bases = lb; res.right_bases = rb;
    res.widths = widths; res.width_heights = wh; res.left_ips = lips; res.right_ips = rips;

    if(Iplat.has_min || Iplat.has_max){
        left_edges  = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)K));
        right_edges = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)K));
        int32_t* plats = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)K));
        for(int32_t i=0;i<K;++i){
            left_edges[i]=LL[i]; right_edges[i]=RR[i]; plats[i]=RR[i]-LL[i]+1;
        }
        res.left_edges = left_edges; res.right_edges = right_edges; res.plateau_sizes = plats;
        if(plateau_sizes) free(plateau_sizes);
    }

    free(P); free(LL); free(RR); free(keep_idx);
    return res;
}

void free_peak_result(peak_result_t* r){
    if(!r) return;
    free(r->indices);
    free(r->peak_heights);
    free(r->left_thresholds); free(r->right_thresholds);
    free(r->prominences); free(r->left_bases); free(r->right_bases);
    free(r->widths); free(r->width_heights); free(r->left_ips); free(r->right_ips);
    free(r->left_edges); free(r->right_edges); free(r->plateau_sizes);
    memset(r, 0, sizeof(*r));
}

// ============================ medfilt (N-D) ============================

static uint32_t prod_u32(const int32_t* a, int32_t n){
    uint64_t p=1; // compute in 64-bit then clamp (defensive)
    for(int32_t i=0; i<n; ++i) p *= (uint64_t)(uint32_t)a[i];
    if(p > 0xFFFFFFFFu) { fprintf(stderr,"medfilt_nd: product overflow\n"); exit(1); }
    return (uint32_t)p;
}

static void unravel_index_u32(uint32_t idx, const int32_t* dims, int32_t nd, int32_t* coords){
    for(int32_t d=nd-1; d>=0; --d){
        int32_t s = dims[d];
        coords[d] = (int32_t)(idx % (uint32_t)s);
        idx /= (uint32_t)s;
    }
}

static uint32_t ravel_index_u32(const int32_t* coords, const int32_t* dims, int32_t nd){
    uint64_t idx=0;
    for(int32_t d=0; d<nd; ++d){
        idx = idx * (uint64_t)(uint32_t)dims[d] + (uint64_t)(uint32_t)coords[d];
        if(idx > 0xFFFFFFFFu){ fprintf(stderr,"medfilt_nd: index overflow\n"); exit(1); }
    }
    return (uint32_t)idx;
}

static int cmp_floats(const void* a, const void* b){
    float da=*(const float*)a, db=*(const float*)b;
    return (da>db)-(da<db);
}

void medfilt_nd(const float* volume, const int32_t* dims, int32_t nd,
                const int32_t* kernel, float* out){
    for(int32_t d=0; d<nd; ++d){
        if(kernel[d]%2 != 1){
            fprintf(stderr, "kernel sizes must be odd\n"); return;
        }
        if(dims[d] <= 0){ fprintf(stderr, "invalid dims\n"); return; }
    }
    uint32_t total = prod_u32(dims, nd);

    int32_t* coords = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)nd));
    int32_t* half   = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)nd));
    for(int32_t d=0; d<nd; ++d) half[d] = kernel[d]/2;

    uint64_t win_sz64 = 1;
    for(int32_t d=0; d<nd; ++d) win_sz64 *= (uint64_t)(uint32_t)kernel[d];
    if(win_sz64 > 0xFFFFFFFFu){ fprintf(stderr,"kernel window too large\n"); exit(1); }
    uint32_t win_size = (uint32_t)win_sz64;

    float* window = (float*)xmalloc((uint32_t)(sizeof(float)*win_size));
    int32_t* kcoord = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)nd));
    int32_t* pos    = (int32_t*)xmalloc((uint32_t)(sizeof(int32_t)*(uint32_t)nd));

    for(uint32_t idx=0; idx<total; ++idx){
        unravel_index_u32(idx, dims, nd, coords);

        // Gather window with zero padding
        uint32_t w=0;
        for(uint32_t wi=0; wi<win_size; ++wi){
            uint32_t t = wi;
            for(int32_t d=nd-1; d>=0; --d){
                int32_t span = kernel[d];
                kcoord[d] = (int32_t)(t % (uint32_t)span) - half[d];
                t /= (uint32_t)span;
            }
            bool inb = true;
            for(int32_t d=0; d<nd; ++d){
                int32_t cd = coords[d] + kcoord[d];
                pos[d] = cd;
                if(cd < 0 || cd >= dims[d]){ inb = false; break; }
            }
            if(inb){
                uint32_t ridx = ravel_index_u32(pos, dims, nd);
                window[w++] = volume[ridx];
            }else{
                window[w++] = 0.0f;
            }
        }

        qsort(window, (size_t)w, sizeof(float), cmp_floats);
        float med = ( (w & 1u) ? window[w/2] : 0.5f*(window[w/2 - 1] + window[w/2]) );
        out[idx] = med;
    }

    free(window); free(kcoord); free(pos);
    free(coords); free(half);
}

