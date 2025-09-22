// numpy_compat.h
#ifndef NUMPY_COMPAT_H
#define NUMPY_COMPAT_H
#include <complex.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { FFT_NORM_BACKWARD = 0, FFT_NORM_FORWARD = 1, FFT_NORM_ORTHO = 2 } fft_norm_t;

fft_norm_t parse_fft_norm(const char* norm);

float* np_fft_rfftfreq(int32_t n, float d);

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

void medfilt_nd(const float* volume, const int32_t* dims, int32_t nd,
                const int32_t* kernel, float* out);

#ifdef __cplusplus
}
#endif
#endif
