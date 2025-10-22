#include "features.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <complex.h>
#include <stdint.h>

// Domain constants (these lived here before; keep them central)
#define MAX_POINTS  508
#define PRE_END     18
#define FULL_END    167
#define POST_START  19
#define RECOVERY_END 20
#define START_END    5

// ---------- local helpers (domain-specific; keep static) ----------
static void slice_segments(const float *fvalues, int16_t n,
                           float pre_seg[PRE_END],
                           float full_seg[FULL_END],
                           float post_seg[MAX_POINTS - POST_START],
                           float rec_seg[RECOVERY_END],
                           float start_seg[START_END],
                           int *post_len_out)
{
    for (int i = 0; i < PRE_END;  i++) pre_seg[i]  = (i < n) ? fvalues[i] : NAN;
    for (int i = 0; i < FULL_END; i++) full_seg[i] = (i < n) ? fvalues[i] : NAN;
    for (int i = 0; i < n - POST_START; i++) post_seg[i] = (i + POST_START < n) ? fvalues[i + POST_START] : NAN;
    for (int i = 0; i < RECOVERY_END; i++)  rec_seg[i]  = (i < n) ? fvalues[i] : NAN;
    for (int i = 0; i < START_END; i++)     start_seg[i]= (i < n) ? fvalues[i] : NAN;
    *post_len_out = (n > POST_START) ? (n - POST_START) : 0;
}

static void compute_post_max(const float *fvalues, int16_t n, int post_len,
                             float *max_post_out, int *max_idx_out)
{
    float max_post = NAN; int max_idx = -1;
    if (post_len > 0) {
        float mv = -FLT_MAX; int seen = 0;
        for (int i = POST_START; i < n; ++i) {
            float v = fvalues[i];
            if (!isnan(v)) { if (!seen || v > mv) { mv = v; max_post = v; max_idx = i; } seen = 1; }
        }
        if (!seen) { max_post = NAN; max_idx = -1; }
    }
    *max_post_out = max_post; *max_idx_out = max_idx;
}

static int argmin_valid_window(const float *x, int len){
    int idx=-1; float mv=FLT_MAX;
    for(int i=0;i<len;i++){ float v=x[i]; if(!isnan(v) && v<mv){ mv=v; idx=i; } }
    return idx;
}

static void first_and_second_derivs_pre(const float full_seg[FULL_END],
                                        float dV_pre[PRE_END],
                                        float d2V_pre[PRE_END],
                                        float *max_rise, float *max_fall,
                                        float *mean_abs_slope, float *std_abs_slope,
                                        float *mean_abs_accel, float *max_accel, float *min_accel)
{
    float dV[FULL_END];
    for (int i = 0; i < FULL_END - 1; i++) {
        float a = full_seg[i], b = full_seg[i+1];
        dV[i] = (!isnan(a) && !isnan(b)) ? (b - a) : NAN;
    }
    dV[FULL_END - 1] = NAN;

    for (int i = 0; i < PRE_END; i++) dV_pre[i] = (i < FULL_END) ? dV[i] : NAN;

    float absdV_pre[PRE_END];
    for (int i = 0; i < PRE_END; i++) absdV_pre[i] = !isnan(dV_pre[i]) ? fabsf(dV_pre[i]) : NAN;

    *max_rise       = nanmax(dV_pre, PRE_END);
    *max_fall       = nanmin(dV_pre, PRE_END);
    *mean_abs_slope = nanmean(absdV_pre, PRE_END);
    *std_abs_slope  = nanstd(absdV_pre, PRE_END);

    float d2V[FULL_END];
    for (int i = 0; i < FULL_END - 2; i++) {
        float a = dV[i], b = dV[i+1];
        d2V[i] = (!isnan(a) && !isnan(b)) ? (b - a) : NAN;
    }
    d2V[FULL_END - 2] = NAN;
    d2V[FULL_END - 1] = NAN;

    for (int i = 0; i < PRE_END; i++) d2V_pre[i] = (i < FULL_END) ? d2V[i] : NAN;

    float absd2V_pre[PRE_END];
    for (int i = 0; i < PRE_END; i++) absd2V_pre[i] = !isnan(d2V_pre[i]) ? fabsf(d2V_pre[i]) : NAN;

    *mean_abs_accel = nanmean(absd2V_pre, PRE_END);
    *max_accel      = nanmax(d2V_pre, PRE_END);
    *min_accel      = nanmin(d2V_pre, PRE_END);
}

static float integrate_area_0_200ms(const float rec_seg[RECOVERY_END]){
    float acc=0.0f; for(int i=0;i<RECOVERY_END;i++){ float v=rec_seg[i]; if(!isnan(v)) acc+=v; }
    return acc * 10.0f;
}

static void count_below_thresholds(const float *fvalues, int16_t n, int *c7, int *c9, int *c10){
    int a=0,b=0,c=0;
    for (int i=0;i<n;i++){ float v=fvalues[i]; if(isnan(v)) continue; if(v<7.0f) a++; if(v<9.0f) b++; if(v<10.0f) c++; }
    *c7=a; *c9=b; *c10=c;
}

static inline float recovery_time_ms_numpy(const float *x, int n, float plus_thresh){
    if(n<=0) return NAN;
    int i0 = nanargmin_first_n(x, n);
    if(i0 < 0) return NAN;
    const float base = x[i0], target = base + plus_thresh;
    for (int j=i0; j<n; ++j){ float v=x[j]; if(v >= target) return (float)((j - i0) * 10); }
    return (float)((n - i0) * 10);
}

// ---------- public API ----------
void compute_features(const float fvalues[], int16_t n, Record *out, float voltage, int16_t measured)
{
    float pre_seg[PRE_END], full_seg[FULL_END], post_seg[MAX_POINTS-POST_START],
          rec_seg[RECOVERY_END], start_seg[START_END];
    int post_len=0;
    slice_segments(fvalues, n, pre_seg, full_seg, post_seg, rec_seg, start_seg, &post_len);

    float min_pre  = nanmin(pre_seg, PRE_END);
    float min_full = nanmin(full_seg, FULL_END);
    float max_full = nanmax(full_seg, FULL_END);
    float std_full = nanstd(full_seg, FULL_END);
    float mean_full= nanmean(full_seg, FULL_END);
    float med_full = nanmedian(full_seg, FULL_END);

    float max_post=NAN; int max_idx=-1;
    compute_post_max(fvalues, n, post_len, &max_post, &max_idx);

    float bounce_back = (!isnan(max_post) && !isnan(min_pre)) ? (max_post - min_pre) : NAN;
    float drop = (!isnan(min_pre) && n>0 && !isnan(fvalues[0])) ? (fvalues[0] - min_pre) : NAN;

    int min_idx = argmin_valid_window(pre_seg, PRE_END);

    float slope_drop = (min_idx > 0 && !isnan(drop)) ? (drop / (-(float)min_idx)) : NAN;
    float slope_bounce_back =
        (!isnan(bounce_back) && min_idx >= 0 && max_idx > min_idx)
        ? (bounce_back / (float)(max_idx - min_idx)) : NAN;

    float start_voltage = nanmean(start_seg, START_END);

    float curve_kurt = nan_kurtosis(full_seg, FULL_END);
    float curve_skew = nan_skew(full_seg, FULL_END);

    float dV_pre[PRE_END], d2V_pre[PRE_END];
    float max_rise_rate_0_180, max_fall_rate_0_180, mean_abs_slope_0_180, std_slope_0_180;
    float mean_abs_accel_0_180, max_accel_0_180, min_accel_0_180;

    first_and_second_derivs_pre(full_seg, dV_pre, d2V_pre,
        &max_rise_rate_0_180, &max_fall_rate_0_180,
        &mean_abs_slope_0_180, &std_slope_0_180,
        &mean_abs_accel_0_180, &max_accel_0_180, &min_accel_0_180);

    int i0 = nanargmin_first_n(fvalues, n);
    float recovery_time_ms;
    if (i0 < 0) {
        recovery_time_ms = NAN;
    } else {
        int stop = first_nan_after_i0_f32(fvalues, n, i0);
        int n_eff = (stop >= 0 ? stop : n);
        recovery_time_ms = recovery_time_ms_numpy(fvalues, n_eff, 0.5f);
    }

    float area_0_200ms = integrate_area_0_200ms(rec_seg);

    int count_below7=0, count_below9=0, count_below10=0;
    count_below_thresholds(fvalues, n, &count_below7, &count_below9, &count_below10);

    float norm_energy_200ms = (measured != 0) ? (area_0_200ms / measured) : NAN;

    float I_est = (measured == 0) ? NAN : measured / 12.0f;
    float r_est = (I_est != 0.0f && !isnan(I_est)) ? (drop / I_est) : NAN;

    float rec_slope = NAN;
    if (!isnan(bounce_back) && !isnan(recovery_time_ms)) {
        int32_t rt_i32 = (int32_t)recovery_time_ms;
        if (rt_i32 != 0) rec_slope = (float)bounce_back / (float)rt_i32;
    }

    const float dt_ms = 10.0f;
    int32_t spike_cnt=0, dip_cnt=0, step_count_sust=0;
    float prom_sum=0.0f, spike_w_mean_ms=0.0f, max_step_mag=0.0f;
    float bp_low=0.0f, bp_mid=0.0f, bp_high=0.0f, bp_mid_ratio=0.0f, bp_high_ratio=0.0f;

    // Residual/peaks/steps/FFT bandpowers (compose from numpy_functions)
    {
        // trend via median filter
        int32_t k_med = (n/50); if(k_med<5) k_med=5; if((k_med & 1)==0) k_med++;
        float* trend = (float*)malloc((size_t)n * sizeof(float));
        int32_t dims[1]={n}, ker[1]={k_med};
        medfilt_nd((const float*)full_seg, dims, 1, ker, trend);

        float* resid = (float*)malloc((size_t)n * sizeof(float));
        for(int i=0;i<n;i++) resid[i] = (i<FULL_END ? full_seg[i] : NAN) - trend[i];
        float sig_res = mad_sigma_f32(resid, n); // from numpy_functions.c

        int32_t min_dist  = (int32_t)fmaxf(2.0f, floorf(30.0f / dt_ms + 1e-6f));
        int32_t min_width = (int32_t)fmaxf(1.0f, floorf(10.0f / dt_ms + 1e-6f));
        float   min_prom  = 3.0f * sig_res;

        float Pmin[2] = { min_prom, NAN };
        float Wmin[2] = { (float)min_width, NAN };

        peak_result_t pos = find_peaks_ref(resid, n, NULL, NULL, min_dist, Pmin, Wmin, 0, 0.5f, NULL);

        float* neg = (float*)malloc((size_t)n * sizeof(float));
        for(int i=0;i<n;i++) neg[i] = -resid[i];
        peak_result_t negp = find_peaks_ref(neg, n, NULL, NULL, min_dist, Pmin, Wmin, 0, 0.5f, NULL);

        spike_cnt = pos.n_peaks + negp.n_peaks;
        dip_cnt   = negp.n_peaks;

        if(pos.prominences) for(int32_t i=0;i<pos.n_peaks;++i) prom_sum += pos.prominences[i];
        if(negp.prominences)for(int32_t i=0;i<negp.n_peaks;++i) prom_sum += negp.prominences[i];

        if(spike_cnt > 0){
            int32_t count=0; float accw=0.0f;
            if(pos.widths)  for(int32_t i=0;i<pos.n_peaks; ++i){ accw += pos.widths[i];  ++count; }
            if(negp.widths) for(int32_t i=0;i<negp.n_peaks;++i){ accw += negp.widths[i]; ++count; }
            if(count>0){ float w_mean = accw/(float)count; spike_w_mean_ms = w_mean * dt_ms; }
        }

        // Sustained steps via boxcar
        int32_t W = (int32_t)fmaxf(4.0f, floorf(40.0f / dt_ms + 1e-6f));
        if(n >= 3*W){
            float* box=(float*)malloc((size_t)W*sizeof(float));
            for(int i=0;i<W;i++) box[i]=1.0f/(float)W;
            int32_t mlen = n - W + 1;
            float* m1=(float*)malloc((size_t)mlen*sizeof(float));
            conv_valid_f32((const float*)full_seg, n, box, W, m1);
            int32_t nsteps = n - 3*W + 2;
            float* steps=(float*)malloc((size_t)nsteps*sizeof(float));
            for(int i=0;i<nsteps;i++){ float a=m1[(2*W-1)+i]; float b=m1[(W-1)+i]; steps[i]=a-b; }

            float step_sig = mad_sigma_f32(steps, nsteps);
            float thr = 4.0f * step_sig;
            for(int i=0;i<nsteps;i++){ float v=fabsf(steps[i]); if(v>thr) step_count_sust++; if(v>max_step_mag) max_step_mag=v; }
            free(steps); free(m1); free(box);
        }

        // Bandpowers
        float* Xr=(float*)malloc(sizeof(float)*(size_t)n);
        float mu = mean_f32(resid, n);
        for(int i=0;i<n;i++) Xr[i] = resid[i] - mu;

        float _Complex* cx=(float _Complex*)malloc(sizeof(float _Complex)*(size_t)n);
        for(int i=0;i<n;i++) cx[i] = Xr[i] + 0.0f*I;
        float _Complex* X = np_fft_fft(cx, n, n, FFT_NORM_BACKWARD, NULL);

        int32_t N = (n%2==0) ? (n/2+1) : ((n+1)/2);
        float* Fr=(float*)malloc(sizeof(float)*(size_t)N);
        for(int k=0;k<N;k++){ float re=crealf(X[k]), im=cimagf(X[k]); Fr[k]=re*re+im*im; }

        float* freqs = np_fft_rfftfreq(n, dt_ms/1000.0f);
        float Ptot = 1e-12f; for(int k=0;k<N;k++) Ptot += Fr[k];
        for(int k=0;k<N;k++){
            float fHz=freqs[k], v=Fr[k];
            if      (fHz>=0.5f && fHz<2.0f)  bp_low  += v;
            else if (fHz>=2.0f && fHz<8.0f)  bp_mid  += v;
            else if (fHz>=8.0f && fHz<20.0f) bp_high += v;
        }
        bp_low/=Ptot; bp_mid/=Ptot; bp_high/=Ptot;
        float denom = bp_low + 1e-12f;
        bp_mid_ratio = bp_mid/denom; bp_high_ratio = bp_high/denom;

        free(freqs); free(Fr); free(X); free(cx); free(Xr);
        free(neg); free_peak_result(&negp); free_peak_result(&pos);
        free(resid); free(trend);
    }

    // write out
    out->voltage = voltage;
    out->measured = measured;
    out->min_val = min_full;
    out->max_val = max_full;
    out->std_dev = std_full;
    out->avg = mean_full;
    out->median = med_full;
    out->bounce_back = bounce_back;
    out->drop = drop;
    out->slope_bounce_back = slope_bounce_back;
    out->slope_drop = slope_drop;
    out->min_volt_below_19 = min_pre;
    out->max_volt_19_above = max_post;
    out->start_voltage = start_voltage;
    out->recovery_time_ms = recovery_time_ms;
    out->area_0_200ms = area_0_200ms;
    out->count_below7 = count_below7;
    out->count_below9 = count_below9;
    out->count_below10 = count_below10;
    out->curve_kurtosis = curve_kurt;
    out->curve_skew = curve_skew;
    out->max_rise_rate_0_180 = max_rise_rate_0_180;
    out->max_fall_rate_0_180 = max_fall_rate_0_180;
    out->mean_abs_slope_0_180 = mean_abs_slope_0_180;
    out->std_slope_0_180 = std_slope_0_180;
    out->mean_abs_accel_0_180 = mean_abs_accel_0_180;
    out->max_accel_0_180 = max_accel_0_180;
    out->min_accel_0_180 = min_accel_0_180;
    out->norm_energy_200ms = norm_energy_200ms;
    out->rec_slope = rec_slope;
    out->r_est = r_est;
    out->spike_cnt = spike_cnt;
    out->dip_cnt = dip_cnt;
    out->prom_sum = prom_sum;
    out->spike_w_mean_ms = spike_w_mean_ms;
    out->step_count_sust = step_count_sust;
    out->max_step_mag = max_step_mag;
    out->bp_low = bp_low;
    out->bp_mid = bp_mid;
    out->bp_high = bp_high;
    out->bp_mid_ratio = bp_mid_ratio;
    out->bp_high_ratio = bp_high_ratio;
}
