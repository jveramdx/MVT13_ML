#ifndef RECORD_H_
#define RECORD_H_

#include <stdint.h>

typedef struct {
    float voltage;
    float measured;
    float min_val;
    float max_val;
    float std_dev;
    float avg;
    float median;
    float bounce_back;
    float drop;
    float slope_bounce_back;
    float slope_drop;
    float min_volt_below_19;
    float max_volt_19_above;
    float start_voltage;
    float time_to_min_ms;
    float recovery_time_ms;
    float area_0_200ms;
    float count_below7;
    float count_below9;
    float count_below10;
    float curve_kurtosis;
    float curve_skew;
    float max_rise_rate_0_180;
    float max_fall_rate_0_180;
    float mean_abs_slope_0_180;
    float std_slope_0_180;
    float mean_abs_accel_0_180;
    float max_accel_0_180;
    float min_accel_0_180;
    float norm_energy_200ms;
    float rec_slope;
    float r_est;
    // spikes/dips
    int32_t spike_cnt;
    int32_t dip_cnt;
    float   prom_sum;
    float   spike_w_mean_ms;
    // sustained steps
    int32_t step_count_sust;
    float   max_step_mag;
    // bandpowers
    float   bp_low;        // [0.5, 2.0) / total
    float   bp_mid;        // [2.0, 8.0) / total
    float   bp_high;       // [8.0, 20.0) / total
    float   bp_mid_ratio;  // bp_mid / (bp_low + 1e-12)
    float   bp_high_ratio; // bp_high / (bp_low + 1e-12)
} Record;

#endif // RECORD_H_
