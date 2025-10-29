#include "features.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <complex.h>
#include <stdint.h>
#include "numpy_functions.h"

// ---------- local helpers (domain-specific; keep available for testing) ----------
void slice_segments(const float *fvalues, int16_t n,
                    float pre_seg[FEATURES_PRE_END],
                    float full_seg[FEATURES_FULL_END],
                    float post_seg[FEATURES_MAX_POINTS - FEATURES_POST_START],
                    float rec_seg[FEATURES_RECOVERY_END],
                    float start_seg[FEATURES_START_END],
                    int *post_len_out)
{
    for (int i = 0; i < FEATURES_PRE_END;  i++) pre_seg[i]  = (i < n) ? fvalues[i] : NAN;
    for (int i = 0; i < FEATURES_FULL_END; i++) full_seg[i] = (i < n) ? fvalues[i] : NAN;
    for (int i = 0; i < n - FEATURES_POST_START; i++) post_seg[i] = (i + FEATURES_POST_START < n) ? fvalues[i + FEATURES_POST_START] : NAN;
    for (int i = 0; i < FEATURES_RECOVERY_END; i++)  rec_seg[i]  = (i < n) ? fvalues[i] : NAN;
    for (int i = 0; i < FEATURES_START_END; i++)     start_seg[i]= (i < n) ? fvalues[i] : NAN;
    *post_len_out = (n > FEATURES_POST_START) ? (n - FEATURES_POST_START) : 0;
}

void compute_post_max(const float *fvalues, int16_t n, int post_len,
                      float *max_post_out, int *max_idx_out)
{
    float max_post = NAN; int max_idx = -1;
    if (post_len > 0) {
        float mv = -FLT_MAX; int seen = 0;
        for (int i = FEATURES_POST_START; i < n; ++i) {
            float v = fvalues[i];
            if (!isnan(v)) { if (!seen || v > mv) { mv = v; max_post = v; max_idx = i; } seen = 1; }
        }
        if (!seen) { max_post = NAN; max_idx = -1; }
    }
    *max_post_out = max_post; *max_idx_out = max_idx;
}

int argmin_valid_window(const float *x, int len){
    int idx=-1; float mv=FLT_MAX;
    for(int i=0;i<len;i++){ float v=x[i]; if(!isnan(v) && v<mv){ mv=v; idx=i; } }
    return idx;
}

void first_and_second_derivs_pre(const float full_seg[FEATURES_FULL_END],
                                 float dV_pre[FEATURES_PRE_END],
                                 float d2V_pre[FEATURES_PRE_END],
                                 float *max_rise, float *max_fall,
                                 float *mean_abs_slope, float *std_abs_slope,
                                 float *mean_abs_accel, float *max_accel, float *min_accel)
{
    float dV[FEATURES_FULL_END];
    for (int i = 0; i < FEATURES_FULL_END - 1; i++) {
        float a = full_seg[i], b = full_seg[i+1];
        dV[i] = (!isnan(a) && !isnan(b)) ? (b - a) : NAN;
    }
    dV[FEATURES_FULL_END - 1] = NAN;

    for (int i = 0; i < FEATURES_PRE_END; i++) dV_pre[i] = (i < FEATURES_FULL_END) ? dV[i] : NAN;

    float absdV_pre[FEATURES_PRE_END];
    for (int i = 0; i < FEATURES_PRE_END; i++) absdV_pre[i] = !isnan(dV_pre[i]) ? fabsf(dV_pre[i]) : NAN;

    *max_rise       = nanmax(dV_pre, FEATURES_PRE_END);
    *max_fall       = nanmin(dV_pre, FEATURES_PRE_END);
    *mean_abs_slope = nanmean(absdV_pre, FEATURES_PRE_END);
    *std_abs_slope  = nanstd(absdV_pre, FEATURES_PRE_END);

    float d2V[FEATURES_FULL_END];
    for (int i = 0; i < FEATURES_FULL_END - 2; i++) {
        float a = dV[i], b = dV[i+1];
        d2V[i] = (!isnan(a) && !isnan(b)) ? (b - a) : NAN;
    }
    d2V[FEATURES_FULL_END - 2] = NAN;
    d2V[FEATURES_FULL_END - 1] = NAN;

    for (int i = 0; i < FEATURES_PRE_END; i++) d2V_pre[i] = (i < FEATURES_FULL_END) ? d2V[i] : NAN;

    float absd2V_pre[FEATURES_PRE_END];
    for (int i = 0; i < FEATURES_PRE_END; i++) absd2V_pre[i] = !isnan(d2V_pre[i]) ? fabsf(d2V_pre[i]) : NAN;

    *mean_abs_accel = nanmean(absd2V_pre, FEATURES_PRE_END);
    *max_accel      = nanmax(d2V_pre, FEATURES_PRE_END);
    *min_accel      = nanmin(d2V_pre, FEATURES_PRE_END);
}

float integrate_area_0_200ms(const float rec_seg[FEATURES_RECOVERY_END]){
    float acc=0.0f; for(int i=0;i<FEATURES_RECOVERY_END;i++){ float v=rec_seg[i]; if(!isnan(v)) acc+=v; }
    return acc * 10.0f;
}

void count_below_thresholds(const float *fvalues, int16_t n, int *c7, int *c9, int *c10){
    int a=0,b=0,c=0;
    for (int i=0;i<n;i++){ float v=fvalues[i]; if(isnan(v)) continue; if(v<7.0f) a++; if(v<9.0f) b++; if(v<10.0f) c++; }
    *c7=a; *c9=b; *c10=c;
}

float recovery_time_ms_numpy(const float *x, int n, float plus_thresh){
    if(n<=0) return NAN;
    int i0 = nanargmin_first_n(x, n);
    if(i0 < 0) return NAN;
    const float base = x[i0], target = base + plus_thresh;
    for (int j=i0; j<n; ++j){ float v=x[j]; if(v >= target) return (float)((j - i0) * 10); }
    return (float)((n - i0) * 10);
}
// median of a small slice (s has no NaNs)
float median_slice(const float *s, int a, int b) {
    if (b <= a) return NAN;
    int n = b - a;
    float buf[64]; // safe for our small windows (<64)
    if (n > 64) n = 64;
    for (int i=0;i<n;i++) buf[i] = s[a+i];
    // insertion sort for tiny n
    for (int i=1;i<n;i++){ float v=buf[i]; int j=i-1; while(j>=0 && buf[j]>v){ buf[j+1]=buf[j]; j--; } buf[j+1]=v; }
    return (n & 1) ? buf[n/2] : 0.5f*(buf[n/2-1]+buf[n/2]);
}

float rolling_var_mean(const float *s, int n, int W){
    if (W < 1) return 0.0f;
    double acc = 0.0; int m = 0;
    for (int i=0;i<n;i++){
        int a = i - W + 1; if (a < 0) a = 0;
        int b = i + 1;      if (b > n) b = n;
        int len = b - a;
        double mu = 0.0; for (int j=a;j<b;j++) mu += s[j]; mu /= (double)len;
        double v  = 0.0; for (int j=a;j<b;j++){ double d=s[j]-mu; v += d*d; }
        v /= (double)len;
        acc += v; m++;
    }
    return (m>0) ? (float)(acc/(double)m) : 0.0f;
}

float variance_f32(const float *x, int n){
    if (n<=0) return 0.0f;
    double mu=0.0; for(int i=0;i<n;i++) mu += x[i]; mu/= (double)n;
    double v=0.0; for(int i=0;i<n;i++){ double d=x[i]-mu; v += d*d; }
    return (float)(v/(double)n);
}

float entropy_e(const float *p, int n){
    double H=0.0;
    for (int i=0;i<n;i++){
        double q = (p[i] <= 0.0f) ? 0.0 : p[i];
        if (q>0.0) H += -q * log(q);
    }
    return (float)H;
}

int longest_true_run(const unsigned char *mask, int n){
    int best=0, cur=0;
    for (int i=0;i<n;i++){
        if (mask[i]) { cur++; if (cur>best) best=cur; }
        else cur=0;
    }
    return best;
}

float autocorr_lag1(const float *x, int n){
    if (n < 3) return 0.0f;
    double mu=0.0; for(int i=0;i<n;i++) mu += x[i]; mu /= (double)n;
    double num=0.0, den=0.0;
    for (int i=0;i<n-1;i++){ double a=x[i]-mu, b=x[i+1]-mu; num += a*b; }
    for (int i=0;i<n;i++){ double a=x[i]-mu; den += a*a; }
    if (den<=0.0) return 0.0f;
    return (float)(num/den);
}


// ---------- public API ----------
void compute_features(const float fvalues[], int16_t n, Record *out, float voltage, int16_t measured)
{
    // constants
    const float dt_ms = 10.0f;

    // ----- slices (0-safe, NaN padded) -----
    float pre_seg[FEATURES_PRE_END], full_seg[FEATURES_FULL_END], post_seg[FEATURES_MAX_POINTS-FEATURES_POST_START],
          rec_seg[FEATURES_RECOVERY_END], start_seg[FEATURES_START_END];
    int post_len = 0;
    slice_segments(fvalues, n, pre_seg, full_seg, post_seg, rec_seg, start_seg, &post_len);

    // ----- basic stats (nan-aware) -----
    float min_pre   = nanmin(pre_seg, FEATURES_PRE_END);
    float min_full  = nanmin(full_seg, FEATURES_FULL_END);
    float max_full  = nanmax(full_seg, FEATURES_FULL_END);
    float std_full  = nanstd(full_seg, FEATURES_FULL_END);
    float mean_full = nanmean(full_seg, FEATURES_FULL_END);
    float med_full  = nanmedian(full_seg, FEATURES_FULL_END);

    // max in post region (absolute index)
    float max_post = NAN; int max_idx = -1;
    compute_post_max(fvalues, n, post_len, &max_post, &max_idx);

    // shape deltas
    float bounce_back = (!isnan(max_post) && !isnan(min_pre)) ? (max_post - min_pre) : NAN;
    float drop        = (!isnan(min_pre) && n>0 && !isnan(fvalues[0])) ? (fvalues[0] - min_pre) : NAN;

    // indices & slopes
    int   min_idx = argmin_valid_window(pre_seg, FEATURES_PRE_END);
    float slope_drop =
        (min_idx > 0 && !isnan(drop)) ? (drop / (-(float)min_idx)) : NAN;
    float slope_bounce_back =
        (!isnan(bounce_back) && min_idx >= 0 && max_idx > min_idx)
        ? (bounce_back / (float)(max_idx - min_idx)) : NAN;

    // start window stats
    float start_voltage = nanmean(start_seg, FEATURES_START_END);

    // higher moments
    float curve_kurt = nan_kurtosis(full_seg, FEATURES_FULL_END);
    float curve_skew = nan_skew(full_seg,   FEATURES_FULL_END);

    // ----- derivatives in 0–180 ms -----
    float dV_pre[FEATURES_PRE_END], d2V_pre[FEATURES_PRE_END];
    float max_rise_rate_0_180, max_fall_rate_0_180, mean_abs_slope_0_180, std_slope_0_180;
    float mean_abs_accel_0_180, max_accel_0_180, min_accel_0_180;
    first_and_second_derivs_pre(
        full_seg, dV_pre, d2V_pre,
        &max_rise_rate_0_180, &max_fall_rate_0_180,
        &mean_abs_slope_0_180, &std_slope_0_180,
        &mean_abs_accel_0_180, &max_accel_0_180, &min_accel_0_180
    );

    // ----- recovery time to +0.5V above minimum (NumPy parity) -----
    float recovery_time_ms;
    {
        int i0 = nanargmin_first_n(fvalues, n);
        if (i0 < 0) {
            recovery_time_ms = NAN;
        } else {
            int stop  = first_nan_after_i0_f32(fvalues, n, i0);
            int n_eff = (stop >= 0 ? stop : n);
            recovery_time_ms = recovery_time_ms_numpy(fvalues, n_eff, 0.5f);
        }
    }

    // area in first 200 ms
    float area_0_200ms = integrate_area_0_200ms(rec_seg);

    // counts below thresholds
    int count_below7 = 0, count_below9 = 0, count_below10 = 0;
    count_below_thresholds(fvalues, n, &count_below7, &count_below9, &count_below10);

    // normalized energy & pseudo-resistance
    float norm_energy_200ms = (measured != 0) ? (area_0_200ms / measured) : NAN;
    float I_est = (measured == 0) ? NAN : measured / 12.0f;
    float r_est = (!isnan(I_est) && I_est != 0.0f) ? (drop / I_est) : NAN;

    // recovery slope (bounce_back / integer ms) to match Python’s where-cast
    float rec_slope = NAN;
    if (!isnan(bounce_back) && !isnan(recovery_time_ms)) {
        int32_t rt_i32 = (int32_t)recovery_time_ms;
        if (rt_i32 != 0) rec_slope = (float)bounce_back / (float)rt_i32;
    }

    // ===== robust residual features (parity with Python) =====
    int32_t spike_cnt = 0, dip_cnt = 0, step_count_sust = 0;
    float prom_sum = 0.0f, spike_w_mean_ms = 0.0f, max_step_mag = 0.0f;
    float bp_low = 0.0f, bp_mid = 0.0f, bp_high = 0.0f, bp_mid_ratio = 0.0f, bp_high_ratio = 0.0f;

    // extras (declare outer scope, assign inside block)
    float longest_flat = 0.0f, hf_energy = 0.0f, spectral_entropy = 0.0f, roll_var = 0.0f;
    float edge_start_diff = 0.0f, edge_end_diff = 0.0f, min_drop2 = 0.0f, recovery_slope_ex = 0.0f;
    float poly_resid = 0.0f, segment_slope_var = 0.0f, zero_cross_rate = 0.0f, resid_spectral_entropy = 0.0f;
    float rel_below_frac = 0.0f, rel_below_longest_ms = 0.0f, win_range_max = 0.0f;
    float tail_std = 0.0f, tail_ac1 = 0.0f, crest_factor = 0.0f, line_length = 0.0f, mid_duty_cycle_low = 0.0f;

    {
        // --- build s = nan_to_num(full_seg[0:n]) ---
        int L = n < FEATURES_FULL_END ? n : FEATURES_FULL_END;
        float *s = (float*)malloc(sizeof(float)*(size_t)L);
        for (int i=0;i<L;i++) s[i] = isnan(full_seg[i]) ? 0.0f : full_seg[i];

        // diff & robust sigma
        int ndiff = (L > 1) ? (L - 1) : 1;
        float *diff = (float*)malloc(sizeof(float)*(size_t)ndiff);
        for (int i=0;i<ndiff;i++) diff[i] = (i+1<L) ? (s[i+1]-s[i]) : 0.0f;
        float sigma_d = mad_sigma_f32(diff, ndiff);

        // longest flat (|diff| < 0.25*sigma_d)
        unsigned char *flat_mask = (unsigned char*)malloc((size_t)ndiff);
        for (int i=0;i<ndiff;i++) flat_mask[i] = (unsigned char)(fabsf(diff[i]) < 0.25f * sigma_d);
        longest_flat = (float)longest_true_run(flat_mask, ndiff);

        // HF energy & spectral entropy from |FFT(s - mean)|
        float mu_s = mean_f32(s, L);
        float *s_zm = (float*)malloc(sizeof(float)*(size_t)L);
        for (int i=0;i<L;i++) s_zm[i] = s[i] - mu_s;
        float _Complex *cx = (float _Complex*)malloc(sizeof(float _Complex)*(size_t)L);
        for (int i=0;i<L;i++) cx[i] = s_zm[i] + 0.0f*I;
        float _Complex *SX = np_fft_fft(cx, L, L, FFT_NORM_BACKWARD, NULL);

        float *mag = (float*)malloc(sizeof(float)*(size_t)L);
        double sum_mag = 0.0;
        for (int k=0;k<L;k++){ float re=crealf(SX[k]), im=cimagf(SX[k]); float m=sqrtf(re*re+im*im); mag[k]=m; sum_mag += m; }
        int q = L/4; double hf_sum = 0.0; for (int k=q;k<L;k++) hf_sum += mag[k];
        hf_energy = (float)(hf_sum / (sum_mag + 1e-6));

        spectral_entropy = 0.0f;
        if (sum_mag > 0.0){
            float *p = (float*)malloc(sizeof(float)*(size_t)L);
            for (int k=0;k<L;k++) p[k] = (float)(mag[k] / (sum_mag + 1e-6));
            spectral_entropy = entropy_e(p, L);
            free(p);
        }

        // rolling variance mean (window=10)
        roll_var = rolling_var_mean(s, L, 10);

        // edge deltas
        int m0 = (L < 10 ? L : 10);
        edge_start_diff = fabsf(median_slice(s, 0, m0) - s[0]);
        edge_end_diff   = fabsf(median_slice(s, L - m0, L) - s[L-1]);

        // min drop & recovery slope (bounded lookahead)
        min_drop2 = FLT_MAX; int drop_idx = -1;
        for (int i=0;i<ndiff;i++) if (diff[i] < min_drop2) { min_drop2 = diff[i]; drop_idx = i; }
        if (drop_idx < 0) { min_drop2 = 0.0f; drop_idx = 0; }
        int lookahead = L - drop_idx - 1; if (lookahead < 1) lookahead = 1; if (lookahead > 20) lookahead = 20;
        recovery_slope_ex = (s[drop_idx + lookahead] - s[drop_idx]) / (float)lookahead;

        // poly residual (deg ≤ 2) via simple normal equations
        {
            int N = L;
            int deg = (N >= 3) ? 2 : 1;
            double Sx=0, Sx2=0, Sx3=0, Sx4=0, Sy=0, Sxy=0, Sx2y=0;
            for (int i=0;i<N;i++){
                double x=i, y=s[i];
                double x2=x*x, x3=x2*x, x4=x3*x;
                Sx+=x; Sx2+=x2; Sx3+=x3; Sx4+=x4; Sy+=y; Sxy+=x*y; Sx2y+=x2*y;
            }
            double a0=0,a1=0,a2=0;
            if (deg==1){
                double NN=N, D=(NN*Sx2 - Sx*Sx);
                if (fabs(D)>1e-12){ a1=(NN*Sxy - Sx*Sy)/D; a0=(Sy - a1*Sx)/NN; }
            } else {
                double A[3][3]={{Sx4,Sx3,Sx2},{Sx3,Sx2,Sx},{Sx2,Sx,(double)N}};
                double B[3]={Sx2y,Sxy,Sy};
                double det =
                    A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1]) -
                    A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) +
                    A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
                if (fabs(det)>1e-12){
                    double A0[3][3]={{B[0],A[0][1],A[0][2]},{B[1],A[1][1],A[1][2]},{B[2],A[2][1],A[2][2]}};
                    double A1[3][3]={{A[0][0],B[0],A[0][2]},{A[1][0],B[1],A[1][2]},{A[2][0],B[2],A[2][2]}};
                    double A2[3][3]={{A[0][0],A[0][1],B[0]},{A[1][0],A[1][1],B[1]},{A[2][0],A[2][1],B[2]}};
                    double det0 =
                        A0[0][0]*(A0[1][1]*A0[2][2]-A0[1][2]*A0[2][1]) -
                        A0[0][1]*(A0[1][0]*A0[2][2]-A0[1][2]*A0[2][0]) +
                        A0[0][2]*(A0[1][0]*A0[2][1]-A0[1][1]*A0[2][0]);
                    double det1 =
                        A1[0][0]*(A1[1][1]*A1[2][2]-A1[1][2]*A1[2][1]) -
                        A1[0][1]*(A1[1][0]*A1[2][2]-A1[1][2]*A1[2][0]) +
                        A1[0][2]*(A1[1][0]*A1[2][1]-A1[1][1]*A1[2][0]);
                    double det2 =
                        A2[0][0]*(A2[1][1]*A2[2][2]-A2[1][2]*A2[2][1]) -
                        A2[0][1]*(A2[1][0]*A2[2][2]-A2[1][2]*A2[2][0]) +
                        A2[0][2]*(A2[1][0]*A2[2][1]-A2[1][1]*A2[2][0]);
                    a0 = det2/det; a1 = det1/det; a2 = det0/det;
                }
            }
            poly_resid = 0.0f;
            for (int i=0;i<N;i++){
                double x=i, y=s[i], yhat = (deg==1) ? (a0 + a1*x) : (a0 + a1*x + a2*x*x);
                double d=y - yhat; poly_resid += (float)(d*d);
            }
            poly_resid /= (float)N;
        }

        // segment slope variance (k=4)
        {
            int kseg=4, sc=0;
            float slopes[4];
            int seg_len = (L / kseg);
            for (int i=0;i<kseg;i++){
                int a = i*seg_len;
                int b = (i==kseg-1) ? (L-1) : ((i+1)*seg_len - 1);
                if (b > a) slopes[sc++] = (s[b]-s[a]) / (float)(b-a);
            }
            segment_slope_var = (sc>1)? variance_f32(slopes, sc) : 0.0f;
        }

        // zero-cross rate of diff
        if (ndiff >= 2){
            int zc=0; for (int i=0;i<ndiff-1;i++) if (diff[i]*diff[i+1] < 0.0f) zc++;
            zero_cross_rate = (float)zc / (float)(L>0?L:1);
        }

        // median trend & residual (NumPy-like)
        int32_t k_med = (L/50); if(k_med < 5) k_med = 5; if((k_med & 1) == 0) k_med++;
        float *trend = (float*)malloc(sizeof(float)*(size_t)L);
        { int32_t dims[1]={L}, ker[1]={k_med}; medfilt_nd(s, dims, 1, ker, trend); }
        float *resid = (float*)malloc(sizeof(float)*(size_t)L);
        for (int i=0;i<L;i++) resid[i] = s[i] - trend[i];
        float sig_res = mad_sigma_f32(resid, L);

        // peaks on resid and -resid (≥30 ms distance, ≥10 ms width, ≥3σ prom)
        int32_t min_dist  = (int32_t)fmaxf(2.0f, floorf(30.0f / dt_ms + 1e-6f));
        int32_t min_width = (int32_t)fmaxf(1.0f, floorf(10.0f / dt_ms + 1e-6f));
        float   min_prom  = 3.0f * sig_res;
        float Pmin[2] = { min_prom, NAN };
        float Wmin[2] = { (float)min_width, NAN };
        peak_result_t pos = find_peaks_ref(resid, L, NULL, NULL, min_dist, Pmin, Wmin, 0, 0.5f, NULL);

        float *neg = (float*)malloc(sizeof(float)*(size_t)L);
        for (int i=0;i<L;i++) neg[i] = -resid[i];
        peak_result_t negp = find_peaks_ref(neg, L, NULL, NULL, min_dist, Pmin, Wmin, 0, 0.5f, NULL);

        spike_cnt = pos.n_peaks + negp.n_peaks;
        dip_cnt   = negp.n_peaks;
        prom_sum  = 0.0f;
        if (pos.prominences)  for (int32_t i=0;i<pos.n_peaks;  ++i) prom_sum += pos.prominences[i];
        if (negp.prominences) for (int32_t i=0;i<negp.n_peaks; ++i) prom_sum += negp.prominences[i];

        if (spike_cnt > 0){
            int32_t cnt=0; float accw=0.0f;
            if (pos.widths)  for (int32_t i=0;i<pos.n_peaks;  ++i){ accw += pos.widths[i];  ++cnt; }
            if (negp.widths) for (int32_t i=0;i<negp.n_peaks; ++i){ accw += negp.widths[i]; ++cnt; }
            if (cnt>0) spike_w_mean_ms = (accw/(float)cnt) * dt_ms;
        }

        // sustained steps via boxcar (W≈40 ms)
        step_count_sust = 0; max_step_mag = 0.0f;
        int32_t W = (int32_t)fmaxf(4.0f, floorf(40.0f / dt_ms + 1e-6f));
        if (L >= 3*W){
            float *box = (float*)malloc(sizeof(float)*(size_t)W);
            for (int i=0;i<W;i++) box[i] = 1.0f/(float)W;
            int32_t mlen = L - W + 1;
            float *m1 = (float*)malloc(sizeof(float)*(size_t)mlen);
            conv_valid_f32(s, L, box, W, m1);
            int32_t nsteps = L - 3*W + 2;
            float *steps = (float*)malloc(sizeof(float)*(size_t)nsteps);
            for (int i=0;i<nsteps;i++){ float a=m1[(2*W-1)+i]; float b=m1[(W-1)+i]; steps[i]=a-b; }
            float step_sig = mad_sigma_f32(steps, nsteps);
            float thr = 4.0f * step_sig;
            for (int i=0;i<nsteps;i++){ float v=fabsf(steps[i]); if (v>thr) step_count_sust++; if (v>max_step_mag) max_step_mag=v; }
            free(steps); free(m1); free(box);
        }

        // bandpowers on residual (zero-mean)
        float mu_r = mean_f32(resid, L);
        float *Xr = (float*)malloc(sizeof(float)*(size_t)L);
        for (int i=0;i<L;i++) Xr[i] = resid[i] - mu_r;

        float _Complex *cx2 = (float _Complex*)malloc(sizeof(float _Complex)*(size_t)L);
        for (int i=0;i<L;i++) cx2[i] = Xr[i] + 0.0f*I;
        float _Complex *X = np_fft_fft(cx2, L, L, FFT_NORM_BACKWARD, NULL);

        int32_t N = (L%2==0) ? (L/2 + 1) : ((L + 1)/2);
        float *Fr = (float*)malloc(sizeof(float)*(size_t)N);
        for (int k=0;k<N;k++){ float re=crealf(X[k]), im=cimagf(X[k]); Fr[k]=re*re+im*im; }

        float *freqs = np_fft_rfftfreq(L, dt_ms/1000.0f);
        float Ptot = 1e-12f; for (int k=0;k<N;k++) Ptot += Fr[k];
        bp_low = bp_mid = bp_high = 0.0f;
        for (int k=0;k<N;k++){
            float fHz=freqs[k], v=Fr[k];
            if      (fHz>=0.5f && fHz<2.0f)  bp_low  += v;
            else if (fHz>=2.0f && fHz<8.0f)  bp_mid  += v;
            else if (fHz>=8.0f && fHz<20.0f) bp_high += v;
        }
        bp_low/=Ptot; bp_mid/=Ptot; bp_high/=Ptot;
        float denom = bp_low + 1e-12f;
        bp_mid_ratio  = bp_mid  / denom;
        bp_high_ratio = bp_high / denom;

        // residual spectral entropy
        resid_spectral_entropy = 0.0f;
        if (Ptot > 0.0f){
            float *q = (float*)malloc(sizeof(float)*(size_t)N);
            for (int k=0;k<N;k++) q[k] = Fr[k] / Ptot + 1e-12f;
            resid_spectral_entropy = entropy_e(q, N);
            free(q);
        }

        // relative-below stats (baseline = median first ~10% clamped [10,100])
        int mbase = L/10; if (mbase < 10) mbase = 10; if (mbase > 100) mbase = 100; if (mbase > L) mbase = L;
        float baseline = median_slice(s, 0, mbase);
        float rel_thr  = baseline - 2.0f * sig_res;
        unsigned char *low_mask = (unsigned char*)malloc((size_t)L);
        int low_cnt = 0;
        for (int i=0;i<L;i++){ low_mask[i] = (unsigned char)(s[i] < rel_thr); if (low_mask[i]) low_cnt++; }
        rel_below_frac        = (float)low_cnt / (float)L;
        rel_below_longest_ms  = (float)longest_true_run(low_mask, L) * dt_ms;

        // rolling range max (~100 ms window)
        int wR = (int)fmaxf(5.0f, floorf(100.0f/dt_ms + 1e-6f));
        win_range_max = 0.0f;
        for (int i=0;i<L;i++){
            int a = i - wR + 1; if (a<0) a=0;
            int b = i + 1;      if (b>L) b=L;
            float lo=FLT_MAX, hi=-FLT_MAX;
            for (int j=a;j<b;j++){ float v=s[j]; if (v<lo) lo=v; if (v>hi) hi=v; }
            float range = hi - lo;
            if (range > win_range_max) win_range_max = range;
        }

        // tail stats (last 10% or at least 1 sample)
        int t0 = (int)(0.9f * L); if (t0 < 0) t0 = 0; if (t0 >= L) t0 = L-1;
        int ntail = L - t0; if (ntail < 1) ntail = 1;
        float *tail = s + t0;
        { double mu_t=0.0; for (int i=0;i<ntail;i++) mu_t += tail[i]; mu_t/= (double)ntail;
          double v=0.0; for (int i=0;i<ntail;i++){ double d=tail[i]-mu_t; v+=d*d; }
          tail_std = (float)sqrt(v/(double)ntail); }
        tail_ac1 = autocorr_lag1(tail, ntail);

        // crest factor & line length
        float max_abs = 0.0f, rms_acc = 0.0f;
        for (int i=0;i<L;i++){ float ab=fabsf(s[i]); if (ab>max_abs) max_abs=ab; rms_acc += s[i]*s[i]; }
        float rms = sqrtf(rms_acc/(float)L) + 1e-12f;
        crest_factor = max_abs / rms;
        line_length  = 0.0f; for (int i=0;i<ndiff;i++) line_length += fabsf(diff[i]); line_length /= (float)L;

        // mid duty-cycle low (20%..80%)
        int aMid = (int)(0.2f*L), bMid = (int)(0.8f*L); if (aMid<0) aMid=0; if (bMid>L) bMid=L;
        int midN = bMid - aMid; if (midN < 1) midN = 1;
        int midLow = 0; for (int i=aMid;i<bMid;i++) if (s[i] < rel_thr) midLow++;
        mid_duty_cycle_low = (float)midLow / (float)midN;

        // cleanup
        free(low_mask);
        free(freqs); free(Fr); free(X); free(cx2); free(Xr);
        free(neg); free_peak_result(&negp); free_peak_result(&pos);
        free(resid); free(trend);
        free(mag); free(SX); free(cx); free(s_zm);
        free(flat_mask); free(diff); free(s);
    }

    // ===== write outputs =====
    out->voltage = voltage;                out->measured = measured;
    out->min_val = min_full;               out->max_val = max_full;
    out->std_dev = std_full;               out->avg = mean_full;
    out->median  = med_full;               out->bounce_back = bounce_back;
    out->drop = drop;                      out->slope_bounce_back = slope_bounce_back;
    out->slope_drop = slope_drop;          out->min_volt_below_19 = min_pre;
    out->max_volt_19_above = max_post;     out->start_voltage = start_voltage;
    out->time_to_min_ms = (min_idx >= 0) ? (float)(min_idx * dt_ms) : NAN;
    out->recovery_time_ms = recovery_time_ms;
    out->area_0_200ms = area_0_200ms;      out->count_below7 = count_below7;
    out->count_below9 = count_below9;      out->count_below10 = count_below10;
    out->curve_kurtosis = curve_kurt;      out->curve_skew = curve_skew;
    out->max_rise_rate_0_180 = max_rise_rate_0_180;
    out->max_fall_rate_0_180 = max_fall_rate_0_180;
    out->mean_abs_slope_0_180 = mean_abs_slope_0_180;
    out->std_slope_0_180 = std_slope_0_180;
    out->mean_abs_accel_0_180 = mean_abs_accel_0_180;
    out->max_accel_0_180 = max_accel_0_180; out->min_accel_0_180 = min_accel_0_180;
    out->norm_energy_200ms = norm_energy_200ms; out->rec_slope = rec_slope;
    out->r_est = r_est;

    // spike/dip & steps & bandpowers (Python parity)
    out->spike_cnt = spike_cnt;            out->dip_cnt = dip_cnt;
    out->prom_sum = prom_sum;              out->spike_w_mean_ms = spike_w_mean_ms;
    out->step_count_sust = step_count_sust; out->max_step_mag = max_step_mag;
    out->bp_low = bp_low;                  out->bp_mid = bp_mid;  out->bp_high = bp_high;
    out->bp_mid_ratio = bp_mid_ratio;      out->bp_high_ratio = bp_high_ratio;

    // extras (exact matches to Python naming/order in packer)
    out->longest_flat = longest_flat;      out->hf_energy = hf_energy;
    out->spectral_entropy = spectral_entropy; out->roll_var = roll_var;
    out->edge_start_diff = edge_start_diff; out->edge_end_diff = edge_end_diff;
    out->min_drop = min_drop2;             out->recovery_slope = recovery_slope_ex;
    out->poly_resid = poly_resid;          out->segment_slope_var = segment_slope_var;
    out->zero_cross_rate = zero_cross_rate;
    out->resid_spectral_entropy = resid_spectral_entropy;
    out->rel_below_frac = rel_below_frac;  out->rel_below_longest_ms = rel_below_longest_ms;
    out->win_range_max = win_range_max;    out->tail_std = tail_std;
    out->tail_ac1 = tail_ac1;              out->crest_factor = crest_factor;
    out->line_length = line_length;        out->mid_duty_cycle_low = mid_duty_cycle_low;
}


