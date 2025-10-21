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

/* === Feature function prototypes added for MVT feature extraction === */
/* Note: FFT-based features (spectral entropy, bandpower, resid_spectral_entropy)
    are intentionally left as stubs or return NAN in C to avoid mismatches with
    NumPy/PocketFFT differences. Time-domain features are implemented here. */

int32_t compute_spike_count(const float* resid, int32_t n);
int32_t compute_dip_count(const float* resid, int32_t n);
float compute_spike_prom_sum(const float* resid, int32_t n);
float compute_spike_width_mean_ms(const float* resid, int32_t n, float dt_ms);

int32_t compute_longest_flat(const float* x, int32_t n, float tol);
float compute_hf_energy(const float* x, int32_t n);
float compute_spectral_entropy_stub(const float* x, int32_t n); /* FFT stub */
float compute_roll_var(const float* x, int32_t n, int32_t window);

float compute_edge_start_diff(const float* x, int32_t n, int32_t window);
float compute_edge_end_diff(const float* x, int32_t n, int32_t window);
float compute_min_drop(const float* x, int32_t n);
float compute_recovery_slope(const float* x, int32_t n, float plus_thresh, float dt_ms);
float compute_poly_resid(const float* x, int32_t n, int32_t degree);
float compute_segment_slope_var(const float* x, int32_t n, int32_t seg_len);
float compute_zero_cross_rate(const float* x, int32_t n);

int32_t compute_step_count_sustained(const float* s, int32_t n, int32_t W);
float compute_max_step_mag(const float* s, int32_t n, int32_t W);

/* Bandpower and resid spectral entropy are FFT-based; provide stubs */
float compute_bp_low_stub(const float* x, int32_t n);
float compute_bp_mid_stub(const float* x, int32_t n);
float compute_bp_high_stub(const float* x, int32_t n);
float compute_resid_spectral_entropy_stub(const float* x, int32_t n);

float compute_rel_below_frac(const float* x, int32_t n, float thr);
int32_t compute_rel_below_longest_ms(const float* x, int32_t n, float thr, float dt_ms);
float compute_win_range_max(const float* x, int32_t n, int32_t win);
float compute_tail_std(const float* x, int32_t n, int32_t tail_len);
float compute_tail_ac1(const float* x, int32_t n, int32_t tail_len);
float compute_crest_factor(const float* x, int32_t n);
float compute_line_length(const float* x, int32_t n);
float compute_mid_duty_cycle_low(const float* x, int32_t n, float low_thr);

#ifdef __cplusplus
}
#endif
#endif
