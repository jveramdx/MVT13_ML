#ifndef NUMPY_FUNCTIONS_H
#define NUMPY_FUNCTIONS_H

#include <complex.h>
#include <stdint.h>

// ---------------- FFT ----------------

typedef enum {
    FFT_NORM_BACKWARD = 0,
    FFT_NORM_FORWARD  = 1,
    FFT_NORM_ORTHO    = 2
} fft_norm_t;

/** Parse "backward" | "forward" | "ortho" into fft_norm_t (BACKWARD on unknown). */
fft_norm_t parse_fft_norm(const char* norm);

/** Real FFT frequency bins like NumPy. len = (n%2==0) ? n/2+1 : (n+1)/2. Caller frees. */
float* np_fft_rfftfreq(int32_t n, float d);

/** Simple reference FFT (DFT) compatible with NumPy semantics for length n. Caller frees. */
float _Complex* np_fft_fft(const float _Complex* x,
                           int32_t n,
                           int32_t out_len,
                           fft_norm_t norm,
                           void* scratch /* unused */);

// ---------------- Peaks / Filters ----------------

typedef struct {
    int32_t* indices; int32_t n_peaks;
    float* peak_heights;
    float* left_thresholds; float* right_thresholds;
    float* prominences; int32_t* left_bases; int32_t* right_bases;
    float* widths; float* width_heights; float* left_ips; float* right_ips;
    int32_t* left_edges; int32_t* right_edges; int32_t* plateau_sizes;
} peak_result_t;

peak_result_t find_peaks_ref(const float* x, int32_t n,
                             const float height[2],
                             const float threshold[2],
                             int32_t distance,
                             const float prominence[2],
                             const float width[2],
                             int32_t wlen,
                             float rel_height,
                             const float plateau_size[2]);

void free_peak_result(peak_result_t* r);

/** N-D median filter with odd kernel per axis (reference, zero-padded). */
void medfilt_nd(const float* volume, const int32_t* dims, int32_t nd,
                const int32_t* kernel, float* out);

// ---------------- Convolution ----------------

/** Valid-mode 1D convolution: out length = nx - nh + 1. */
void conv_valid_f32(const float* x, int32_t nx,
                    const float* h, int32_t nh,
                    float* y /* len = nx - nh + 1 */);

// ---------------- Stats (generic, NaN-aware where named) ----------------

float mean_f32(const float* x, int32_t n);

float nanmin(const float* arr, int n);
float nanmax(const float* arr, int n);
float nanmean(const float* arr, int n);
float nanstd (const float* arr, int n);
float nanmedian(const float* arr, int n);
float nan_kurtosis(const float* arr, int n);
float nan_skew     (const float* arr, int n);
float mad_sigma_f32(const float* x, int32_t n);

// ---------------- Small utilities that mirror NumPy behavior ----------------

/** First index of finite minimum (np.nanargmin); returns -1 if all NaN. */
int32_t nanargmin_first_n(const float* x, int32_t n);

/** First index >= i0 where x[idx] is NaN; returns -1 if none. */
int32_t first_nan_after_i0_f32(const float* x, int32_t n, int32_t i0);

#endif // NUMPY_FUNCTIONS_H
