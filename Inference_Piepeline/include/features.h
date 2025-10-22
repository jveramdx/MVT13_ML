#ifndef FEATURES_H_
#define FEATURES_H_

#include <stdint.h>
#include "record.h"
#include "numpy_functions.h"

// Shared constants for feature computations
enum {
    FEATURES_MAX_POINTS   = 508,
    FEATURES_PRE_END      = 18,
    FEATURES_FULL_END     = 167,
    FEATURES_POST_START   = 19,
    FEATURES_RECOVERY_END = 20,
    FEATURES_START_END    = 5
};

// Segment helpers
void slice_segments(const float *fvalues, int16_t n,
                    float pre_seg[FEATURES_PRE_END],
                    float full_seg[FEATURES_FULL_END],
                    float post_seg[FEATURES_MAX_POINTS - FEATURES_POST_START],
                    float rec_seg[FEATURES_RECOVERY_END],
                    float start_seg[FEATURES_START_END],
                    int *post_len_out);

void compute_post_max(const float *fvalues, int16_t n, int post_len,
                      float *max_post_out, int *max_idx_out);

int argmin_valid_window(const float *x, int len);

void first_and_second_derivs_pre(const float full_seg[FEATURES_FULL_END],
                                 float dV_pre[FEATURES_PRE_END],
                                 float d2V_pre[FEATURES_PRE_END],
                                 float *max_rise, float *max_fall,
                                 float *mean_abs_slope, float *std_abs_slope,
                                 float *mean_abs_accel, float *max_accel, float *min_accel);

float integrate_area_0_200ms(const float rec_seg[FEATURES_RECOVERY_END]);

void count_below_thresholds(const float *fvalues, int16_t n,
                            int *c7, int *c9, int *c10);

float recovery_time_ms_numpy(const float *x, int n, float plus_thresh);

float median_slice(const float *s, int a, int b);

float rolling_var_mean(const float *s, int n, int W);

float variance_f32(const float *x, int n);

float entropy_e(const float *p, int n);

int longest_true_run(const unsigned char *mask, int n);

float autocorr_lag1(const float *x, int n);

/** Compute all engineered features for a single trace. */
void compute_features(const float fvalues[], int16_t n,
                      Record *out, float voltage, int16_t measured);

#endif // FEATURES_H_
