#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <libgen.h>   /* for dirname / basename on POSIX */
#include <sys/stat.h> /* for checking if file exists */
#include "numpy_functions.h"
#include <stdbool.h>

#define MAX_POINTS 508
#define MAX_ALTERNATOR_RIPPLE 257
#define PRE_END 18
#define FULL_END 167
#define POST_START 19
#define RECOVERY_END 20
#define START_END 5

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
    float   bp_low;       // [0.5, 2.0) / total
    float   bp_mid;       // [2.0, 8.0) / total
    float   bp_high;      // [8.0, 20.0) / total
    float   bp_mid_ratio; // bp_mid / (bp_low + 1e-12)
    float   bp_high_ratio;// bp_high / (bp_low + 1e-12)
} Record;

typedef struct {
    const char *version_str;
    int16_t mapped_value;
} SoftwareVersionMap;

typedef struct {
    int idx_id;
    int idx_starter;
    int idx_swver;
    int idx_voltage;
    int idx_measured;
    int idx_ripple;
} ColIdx;


//--------------Utility Functions-------------------
// ---------- utility: qsort comparator for float ----------
static int cmp_f32(const void* a, const void* b){
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa>fb) - (fa<fb);
}

// ---------- median (float32) ----------
static float median_f32(const float* x, int32_t n){
    if(n <= 0) return 0.0f;
    float* tmp = (float*)malloc((size_t)n*sizeof(float));
    memcpy(tmp, x, (size_t)n*sizeof(float));
    qsort(tmp, (size_t)n, sizeof(float), cmp_f32);
    float med = (n & 1) ? tmp[n/2] : 0.5f*(tmp[n/2 - 1] + tmp[n/2]);
    free(tmp);
    return med;
}

// ---------- MAD-based robust sigma (float32) ----------
static float mad_sigma_f32(const float* x, int32_t n){
    if(n <= 0) return 1e-12f;
    float med = median_f32(x, n);
    float* d = (float*)malloc((size_t)n*sizeof(float));
    for(int32_t i=0;i<n;++i) d[i] = fabsf(x[i] - med);
    float mad = median_f32(d, n);
    free(d);
    return 1.4826f * mad + 1e-12f;
}

// ---------- 1D valid-mode convolution y = conv(x, h), valid only ----------
static void conv_valid_f32(const float* x, int32_t nx,
                           const float* h, int32_t nh,
                           float* y /* len = nx - nh + 1 */){
    int32_t ny = nx - nh + 1;
    for(int32_t i=0;i<ny;++i){
        float acc = 0.0f;
        for(int32_t k=0;k<nh;++k) acc += x[i+k] * h[k];
        y[i] = acc;
    }
}

// ---------- mean of float32 (n>0) ----------
static float mean_f32(const float* a, int32_t n){
    float s = 0.0f;
    for(int32_t i=0;i<n;++i) s += a[i];
    return s / (float)n;
}

int hex_to_signed_int(const char *h) {
    // Convert an 8-digit hex string (two’s-complement 32-bit) into a C int
    if (!h) return 0;
    unsigned int val = 0;
    sscanf(h, "%8x", &val);
    if (val & 0x80000000U) {
        return (int)(val - 0x100000000U);
    }
    return (int)val;
}
int compare_floats(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

float convert_value_version(const char *token) {
    if (token == NULL || strlen(token) == 0)
        return NAN;
    char *endptr;
    
    int val = strtol(token, &endptr, 16);
    if (endptr == token) return NAN;
    return (float)val / 100.0f;
    }

int16_t process_starter_voltage_version(const char *input, float values[MAX_POINTS]) {
    int16_t written = 0;
    int16_t last_non_nan = -1;

    // Defensive init: everything is NaN first (so tail stays NaN)
    for (int i = 0; i < MAX_POINTS; i++) values[i] = NAN;

    if (!input) return 0;

    char *copy = strdup(input);
    if (!copy) return 0;

    char *save = NULL;
    char *token = strtok_r(copy, ":", &save);
    while (token && written < MAX_POINTS) {
        float v = convert_value_version(token);   // ensure this returns NAN for "", "nan", etc.
        values[written] = v;
        if (!isnan(v)) last_non_nan = written;   // track last non-NaN index
        written++;
        token = strtok_r(NULL, ":", &save);
    }

    free(copy);

    // n = last non-NaN index + 1; if none, n = 0
    return (int16_t)((last_non_nan >= 0) ? (last_non_nan + 1) : 0);
}

//Write a function that applies the hex_to_signed_int function to convert the alternator ripple graph array(here its all hex to a signed int).

int process_alternator_ripple(const char *input, float values[MAX_ALTERNATOR_RIPPLE])
{
    int16_t count = 0;
    char *copy = strdup(input);
    if (!copy) return 0;
    char *token = strtok(copy, ":");
    while (token != NULL && count < MAX_ALTERNATOR_RIPPLE) {
        values[count++] = hex_to_signed_int(token);
        token = strtok(NULL, ":");
    }
    free(copy);
    return count;
}

float sum_alternator_ripple_array(const float *arr, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

// Helper: nanmin/nanmax/nanmean/nanstd for 1D float arrays
static float nanmin(const float *arr, int n)
{
    float minv = FLT_MAX;
    for (int i = 0; i < n; i++)
    {
        if (!isnan(arr[i]) && arr[i] < minv)
        {
            minv = arr[i];
        }
    }
    return minv;
}
static float nanmax(const float *arr, int n)
{
    float maxv = -FLT_MAX;
    for (int i = 0; i < n; i++)
    {
        if (!isnan(arr[i]) && arr[i] > maxv)
        {
            maxv = arr[i];
        }
    }
    return maxv;
}

static float nanmean(const float *arr, int n)
{
    float sum = 0.0f;
    int cnt = 0;
    for (int i = 0; i < n; i++)
    {
        if (!isnan(arr[i]))
        {
          sum += arr[i]; cnt++;
        }
    }
    return cnt > 0 ? sum / cnt : NAN;
}

static float nanstd(const float *arr, int n)
{
    float m = nanmean(arr, n);
    float sumsq = 0.0f; int cnt = 0;
    for (int i = 0; i < n; i++)
    {
        if (!isnan(arr[i]))
        {
            float d = arr[i] - m;
            sumsq += d * d; cnt++;
        }
    }
    return cnt > 0 ? sqrtf(sumsq / cnt) : NAN;
}

static float nanmedian(float *arr, int n)
{
    int cnt = 0;
    float *tmp = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) if (!isnan(arr[i])) tmp[cnt++] = arr[i];
    float med = NAN;
    if (cnt > 0)
    {
        qsort(tmp, cnt, sizeof(float), compare_floats);
        if (cnt % 2 == 0)
        {
            med = (tmp[cnt/2 - 1] + tmp[cnt/2]) / 2.0f;
        }
        else
        {
            med = tmp[cnt/2];
        }
    }
    free(tmp);
    return med;
}


// Kurtosis and skew (unbiased, Fisher=False, bias=False, nan_policy='omit')
static float nan_kurtosis(const float *arr, int n) {
    // collect moments over non-NaNs
    float m = 0.0f, s2 = 0.0f, s4 = 0.0f;
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        if (!isnan(arr[i])) { m += arr[i]; cnt++; }
    }
    if (cnt < 4) return NAN;
    m /= (float)cnt;

    for (int i = 0; i < n; i++) {
        if (!isnan(arr[i])) {
            float d = (float)arr[i] - m;
            float d2 = d*d;
            s2 += d2;
            s4 += d2*d2;
        }
    }
    float m2 = s2 / (float)cnt;   // population central moments
    float m4 = s4 / (float)cnt;

    // Guard "zero variance" like SciPy (approximate)
    // SciPy checks m2 <= (eps * mean)^2 along axis; this is a close stand-in:
    if (m2 <= DBL_EPSILON * m * m) return NAN;

    float g2 = m4 / (m2 * m2);     // Pearson kurtosis (biased)

    // Unbiased Pearson correction (SciPy):
    // k_unbiased = ((n^2 - 1) * g2 - 3*(n-1)^2) / ((n-2)*(n-3)) + 3
    float nn = (float)cnt;
    float num = (nn*nn - 1.0f) * g2 - 3.0f * (nn - 1.0f) * (nn - 1.0f);
    float den = (nn - 2.0f) * (nn - 3.0f);
    if (den == 0.0) return NAN;
    float k = num / den + 3.0f;

    return (float)k;  // fisher=False → keep Pearson (do NOT subtract 3)
}
static float nan_skew(const float *a, int n_total) {
    // 1) collect finite values stats (nan_policy='omit')
    int n = 0;
    float mean = 0.0;

    for (int i = 0; i < n_total; ++i) {
        float v = a[i];
        if (!isnan(v)) { mean += (float)v; n++; }
    }
    if (n < 3) return NAN;
    mean /= (float)n;

    // 2) biased central moments m2, m3 (divide by N), over finite values
    float s2 = 0.0, s3 = 0.0;
    for (int i = 0; i < n_total; ++i) {
        float vf = a[i];
        if (!isnan(vf)) {
            float d = (float)vf - mean;
            float d2 = d * d;
            s2 += d2;
            s3 += d2 * d;
        }
    }
    float m2 = s2 / (float)n;
    float m3 = s3 / (float)n;

    // 3) SciPy epsilon guard: zero = (m2 <= (eps * mean)**2)
    const float eps = DBL_EPSILON;                 // xp.finfo(m2.dtype).eps with float accumulators
    const float thr = (eps * mean) * (eps * mean);
    const bool zero = (m2 <= thr);
    if (zero) return NAN;

    // 4) g1 and bias=False correction if n > 2
    float g1 = m3 / pow(m2, 1.5);
    if (n > 2) {
        float dn = (float)n;
        float G1 = sqrt(dn * (dn - 1.0)) / (dn - 2.0) * g1;
        return (float)G1;                            // 5) cast to float (like SciPy returning float64 -> you can cast to float32)
    } else {
        return NAN;
    }
}


static inline int nanargmin_first_n(const float *x, int n) {
    int idx = -1;
    float mv = FLT_MAX;   // float32 max
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        if (!isnan(v)) {
            // strictly '<' → first occurrence tie-break like NumPy
            if (v < mv) { mv = v; idx = i; }
        }
    }
    return idx; // -1 → all-NaN
}
static inline int first_nan_after_i0_f32(const float *x, int n, int i0) {
    for (int j = i0; j < n; ++j) if (isnan(x[j])) return j;
    return -1; // no NaN after i0
}

static inline float recovery_time_ms_numpy(const float *x, int n, float plus_thresh) {
    if (n <= 0) return NAN;

    int i0 = nanargmin_first_n(x, n);
    if (i0 < 0) return NAN;  // all-NaN row

    const float base   = x[i0];
    const float target = base + plus_thresh;   // keep in float32

    // scan full row: j = i0 .. n-1
    for (int j = i0; j < n; ++j) {
        float v = x[j];
        // NaN comparison is false; no special handling needed
        if (v >= target) {
            return (float)((j - i0) * 10);    // 10 ms/sample
        }
    }
    // fallback identical to Python
    return (float)((n - i0) * 10);
}


// Main feature computation
void compute_features(const float fvalues[], int16_t n, Record *out, float voltage, int16_t measured) {
    // Slices
    float pre_seg[PRE_END], full_seg[FULL_END], post_seg[MAX_POINTS-POST_START], rec_seg[RECOVERY_END], start_seg[START_END];
    int i;
    for (i = 0; i < PRE_END; i++) pre_seg[i] = (i < n) ? fvalues[i] : NAN;
    for (i = 0; i < FULL_END; i++) full_seg[i] = (i < n) ? fvalues[i] : NAN;
    for (i = 0; i < n-POST_START; i++) post_seg[i] = (i+POST_START < n) ? fvalues[i+POST_START] : NAN;
    for (i = 0; i < RECOVERY_END; i++) rec_seg[i] = (i < n) ? fvalues[i] : NAN;
    for (i = 0; i < START_END; i++) start_seg[i] = (i < n) ? fvalues[i] : NAN;
    const int post_len = (n > POST_START) ? (n - POST_START) : 0;
	// Basic statistics
	float min_pre = nanmin(pre_seg, PRE_END);
	float min_full = nanmin(full_seg, FULL_END);
	float max_full = nanmax(full_seg, FULL_END);
	float std_full = nanstd(full_seg, FULL_END);
	float mean_full = nanmean(full_seg, FULL_END);
	float med_full = nanmedian(full_seg, FULL_END);

	float max_post = NAN;
    int   max_idx  = -1;     // absolute index into fvalues (like NumPy)
    if (post_len > 0) {
        float mv = -FLT_MAX; int seen = 0;
        for (int i = POST_START; i < n; ++i) {
            float v = fvalues[i];
            if (!isnan(v)) {
                if (!seen || v > mv) { mv = v; max_post = v; max_idx = i; }
                seen = 1;
            }
        }
        if (!seen) { max_post = NAN; max_idx = -1; } // all-NaN in post
    }

	float bounce_back = (!isnan(max_post) && !isnan(min_pre)) ? (max_post - min_pre) : NAN;
    float drop = (!isnan(min_pre) && n > 0 && !isnan(fvalues[0])) ? (fvalues[0] - min_pre) : NAN;

    // ---------- Min index in PRE for slope denominators ----------
    int min_idx = -1;
    {
        float mv = FLT_MAX;
        for (int i = 0; i < PRE_END; i++) {
            float v = pre_seg[i];
            if (!isnan(v) && v < mv) { mv = v; min_idx = i; }
        }
    }

    // ---------- Slopes ----------
    float slope_drop = (min_idx > 0 && !isnan(drop)) ? (drop / (-(float)min_idx)) : NAN;
    float slope_bounce_back =
        (!isnan(bounce_back) && min_idx >= 0 && max_idx > min_idx)
        ? (bounce_back / (float)(max_idx - min_idx)) : NAN;
	// Start voltage and time to min
	float start_voltage = nanmean(start_seg, START_END);
	//float time_to_min_ms = (min_idx >= 0) ? min_idx * 10.0f : NAN;

	// Kurtosis and skew
	float curve_kurt = nan_kurtosis(full_seg, FULL_END);
	float curve_skew = nan_skew(full_seg, FULL_END);


    // 1st derivative
    float dV[FULL_END];
    for (i = 0; i < FULL_END - 1; i++) {
        dV[i] = (!isnan(full_seg[i]) && !isnan(full_seg[i+1]))
                 ? full_seg[i+1] - full_seg[i]
                 : NAN;
	}
    dV[FULL_END - 1] = NAN;
    float dV_pre[PRE_END];
    for (i = 0; i < PRE_END; i++){
        dV_pre[i] = (i < FULL_END) ? dV[i] : NAN;
    }
    // compute mean-abs and std of the slope
    float absdV_pre[PRE_END];
    for (i = 0; i < PRE_END; i++){
        absdV_pre[i] = !isnan(dV_pre[i]) ? fabsf(dV_pre[i]) : NAN;
	}
    float max_rise_rate_0_180  = nanmax(dV_pre, PRE_END);
    float max_fall_rate_0_180  = nanmin(dV_pre, PRE_END);
    float mean_abs_slope_0_180 = nanmean(absdV_pre, PRE_END);
    float std_slope_0_180      = nanstd(absdV_pre, PRE_END);

    // 2nd derivative
    float d2V[FULL_END];
    for (i = 0; i < FULL_END - 2; i++){
        d2V[i] = (!isnan(dV[i]) && !isnan(dV[i+1]))
                  ? dV[i+1] - dV[i]
                  : NAN;
    }
    d2V[FULL_END - 2] = NAN;
    d2V[FULL_END - 1] = NAN;
    float d2V_pre[PRE_END];
    for (i = 0; i < PRE_END; i++){
        d2V_pre[i] = (i < FULL_END) ? d2V[i] : NAN;
	}
    float absd2V_pre[PRE_END];
    for (i = 0; i < PRE_END; i++){
		absd2V_pre[i] = !isnan(d2V_pre[i]) ? fabsf(d2V_pre[i]) : NAN;
	}
    float mean_abs_accel_0_180 = nanmean(absd2V_pre, PRE_END);
    float max_accel_0_180      = nanmax(d2V_pre, PRE_END);
    float min_accel_0_180      = nanmin(d2V_pre, PRE_END);

        // --- Recovery time: contiguous finite window [i0, n_eff) ---
    int i0 = nanargmin_first_n(fvalues, n);         // dip index (first min)
    float recovery_time_ms;
    if (i0 < 0) {
        recovery_time_ms = NAN;                     // all-NaN row
    } else {
        int stop = first_nan_after_i0_f32(fvalues, n, i0);
        int n_eff = (stop >= 0 ? stop : n);         // stop at first NaN; else use n
        recovery_time_ms = recovery_time_ms_numpy(fvalues, n_eff, 0.5f);
    }


	// Area 0-200ms
	float area_0_200ms = 0.0f;
	for (i = 0; i < RECOVERY_END; i++)
		{
			if (!isnan(rec_seg[i]))
			{
				area_0_200ms += rec_seg[i];
			}
		}
	area_0_200ms *= 10.0f;

	// Count below thresholds
    int count_below7 = 0, count_below9 = 0, count_below10 = 0;
    for (int i = 0; i < n; i++) {
        float v = fvalues[i];
        if (isnan(v)) continue;          // NaNs don't count, same as NumPy
        if (v <  7.0f) count_below7++;
        if (v <  9.0f) count_below9++;
        if (v < 10.0f) count_below10++;
    }


	// Norm_Energy_200ms
	float norm_energy_200ms = (measured != 0) ? area_0_200ms / measured : NAN;

	// R_est (pseudo-resistance)
	float I_est = (measured == 0) ? NAN : measured / 12.0f;
	float r_est = (I_est != 0.0f && !isnan(I_est)) ? drop / I_est : NAN;

	
    	// Recovery time until +0.5V above min
    float rec_slope = NAN;
    if (!isnan(bounce_back) && !isnan(recovery_time_ms)) {
        // Python's where mask is built on int32(Recovery_Time_ms) != 0
        // Our recovery_time_ms is always an exact multiple of 10.0f, so a cast is safe.
        int32_t rt_i32 = (int32_t)recovery_time_ms;  // matches astype(np.int32)

        if (rt_i32 != 0) {
            // Use float32 operands like Python's astype(np.float32)
            float num = (float)bounce_back;
            float den = (float)rt_i32;               // not recovery_time_ms (avoid any fp wobble)
            rec_slope = num / den;
        } // else leave as NaN (same as np.divide(..., where=rt!=0))
    }

    // Time step in ms
    float dt_ms = 10.0f;


    // ---------------- median-detrended residual ----------------
    int32_t k_med = 5;
    {   // k_med = max(5, (n // 50) | 1)  (odd)
        int32_t km = n / 50;
        if(km < 5) km = 5;
        if((km & 1) == 0) km += 1;
        k_med = km;
    }

    float* s = full_seg;
    // medfilt (1-D) -> median trend
    float* trend = (float*)malloc((size_t)n*sizeof(float));
    {
        int32_t dims[1] = { n };
        int32_t ker[1]  = { k_med };
        medfilt_nd(s, dims, 1, ker, trend);
    }

    // residual = s - trend
    float* resid = (float*)malloc((size_t)n*sizeof(float));
    for(int32_t i=0;i<n;++i) resid[i] = s[i] - trend[i];

    // robust sigma from MAD
    float sig_res = mad_sigma_f32(resid, n);

    // peak conditions
    int32_t min_dist  = (int32_t)fmaxf(2.0f, floorf(30.0f / dt_ms + 1e-6f)); // ≥ 30ms
    int32_t min_width = (int32_t)fmaxf(1.0f, floorf(10.0f / dt_ms + 1e-6f)); // ≥ 10ms
    float   min_prom  = 3.0f * sig_res;

    // find_peaks on resid (positive spikes)
    float NaN = NAN;
    float Pmin[2] = { min_prom, NaN };    // prominence >= min_prom
    float Wmin[2] = { (float)min_width, NaN }; // width >= min_width (samples)

    peak_result_t pos = find_peaks_ref(
        resid, n,
        /*height*/NULL,
        /*threshold*/NULL,
        /*distance*/min_dist,
        /*prominence*/Pmin,
        /*width*/Wmin,
        /*wlen*/0,
        /*rel_height*/0.5f,
        /*plateau*/NULL
    );

    // find_peaks on -resid (negative spikes / dips)
    float* neg = (float*)malloc((size_t)n*sizeof(float));
    for(int32_t i=0;i<n;++i) neg[i] = -resid[i];

    peak_result_t negp = find_peaks_ref(
        neg, n,
        /*height*/NULL,
        /*threshold*/NULL,
        /*distance*/min_dist,
        /*prominence*/Pmin,
        /*width*/Wmin,
        /*wlen*/0,
        /*rel_height*/0.5f,
        /*plateau*/NULL
    );

    // spike counts
    int32_t spike_cnt = pos.n_peaks + negp.n_peaks;
    int32_t dip_cnt   = negp.n_peaks;

    // sum of prominences
    float prom_sum = 0.0f;
    if(pos.prominences){
        for(int32_t i=0;i<pos.n_peaks;++i) prom_sum += pos.prominences[i];
    }
    if(negp.prominences){
        for(int32_t i=0;i<negp.n_peaks;++i) prom_sum += negp.prominences[i];
    }

    // mean width (samples) -> ms
    float spike_w_mean_ms = 0.0f;
    if(spike_cnt > 0){
        int32_t count = 0;
        float accw = 0.0f;
        if(pos.widths){ for(int32_t i=0;i<pos.n_peaks;++i){ accw += pos.widths[i]; ++count; } }
        if(negp.widths){ for(int32_t i=0;i<negp.n_peaks;++i){ accw += negp.widths[i]; ++count; } }
        if(count > 0){
            float w_mean = accw / (float)count;
            spike_w_mean_ms = w_mean * dt_ms;
        }
    }

    // ---------------- sustained steps (boxcar diff of moving averages) --------
    int32_t W = (int32_t)fmaxf(4.0f, floorf(40.0f / dt_ms + 1e-6f)); // ~40 ms window
    int32_t step_count_sust = 0;
    float   max_step_mag = 0.0f;

    if(n >= 3 * W){
        // box = ones(W)/W
        float* box = (float*)malloc((size_t)W*sizeof(float));
        for(int32_t i=0;i<W;++i) box[i] = 1.0f / (float)W;

        // m1 = conv(s, box, valid) -> length n-W+1
        int32_t mlen = n - W + 1;
        float* m1 = (float*)malloc((size_t)mlen*sizeof(float));
        conv_valid_f32(s, n, box, W, m1);

        // steps = m1[2W-1:] - m1[W-1:-W]
        // length = mlen - (2W-1) = n - 3W + 2
        int32_t nsteps = n - 3*W + 2;
        float* steps = (float*)malloc((size_t)nsteps*sizeof(float));
        for(int32_t i=0;i<nsteps;++i){
            float a = m1[(2*W - 1) + i];
            float b = m1[(W - 1) + i];
            steps[i] = a - b;
        }

        // MAD sigma of steps
        float step_sig = mad_sigma_f32(steps, nsteps);
        float thr = 4.0f * step_sig;

        // count |steps| > thr, and track max magnitude
        int32_t cnt = 0;
        float maxmag = 0.0f;
        for(int32_t i=0;i<nsteps;++i){
            float v = fabsf(steps[i]);
            if(v > thr) cnt++;
            if(v > maxmag) maxmag = v;
        }
        step_count_sust = cnt;
        max_step_mag = maxmag;

        free(steps); free(m1); free(box);
    } else {
        step_count_sust = 0;
        max_step_mag = 0.0f;
    }

    // ---------------- residual bandpowers & ratios ----------------------------
    // Xr = resid - mean(resid)
    float* Xr = (float*)malloc((size_t)n*sizeof(float));
    float mu = mean_f32(resid, n);
    for(int32_t i=0;i<n;++i) Xr[i] = resid[i] - mu;

    // rFFT via full FFT of real input -> take first N bins
    int32_t N = (n % 2 == 0) ? (n/2 + 1) : ((n + 1)/2);
    float _Complex* cx = (float _Complex*)malloc((size_t)n*sizeof(float _Complex));
    for(int32_t i=0;i<n;++i) cx[i] = Xr[i] + 0.0f*I;

    float _Complex* X = np_fft_fft(cx, n, n, FFT_NORM_BACKWARD, NULL);

    // power spectrum Fr = |X[k]|^2 for k=0..N-1
    float* Fr = (float*)malloc((size_t)N*sizeof(float));
    for(int32_t k=0;k<N;++k){
        float re = crealf(X[k]), im = cimagf(X[k]);
        Fr[k] = re*re + im*im;
    }

    // frequency bins
    float d = dt_ms / 1000.0f;
    float* freqs = np_fft_rfftfreq(n, d); // length N

    // band helper
    float Ptot = 1e-12f;
    for(int32_t k=0;k<N;++k) Ptot += Fr[k];

    // [0.5, 2.0), [2.0, 8.0), [8.0, 20.0)
    float bp_low = 0.0f, bp_mid = 0.0f, bp_high = 0.0f;
    for(int32_t k=0;k<N;++k){
        float fHz = freqs[k];
        float v   = Fr[k];
        if(fHz >= 0.5f  && fHz < 2.0f)  bp_low  += v;
        else if(fHz >= 2.0f && fHz < 8.0f)  bp_mid  += v;
        else if(fHz >= 8.0f && fHz < 20.0f) bp_high += v;
    }
    bp_low  /= Ptot;
    bp_mid  /= Ptot;
    bp_high /= Ptot;

    float denom = bp_low + 1e-12f;
    float bp_mid_ratio  = bp_mid  / denom;
    float bp_high_ratio = bp_high / denom;

    // Populate output record
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


    // ---------------- cleanup ----------------
    free(freqs);
    free(Fr);
    free(X);
    free(cx);
    free(Xr);

    free(neg);
    free_peak_result(&negp);
    free_peak_result(&pos);

    free(resid);
    free(trend);

}

void preprocess_record(const char *starter_str, const char *software_version_str,
                      float voltage, float measured, Record *out) {
    float fvalues[MAX_POINTS];
    for (int i = 0; i < MAX_POINTS; i++) {
        fvalues[i] = NAN;   // float NaN is the closest thing to "null" in C
    }
    int16_t n = process_starter_voltage_version(starter_str, fvalues);
    compute_features(fvalues, n, out, voltage, measured);
}

// Declare the external model function (from m2cgen export).
void score(const float * input, float * output);
/* ===================== ADDED: feature packing (uses your existing Record) ===================== */
static void pack_features(const Record *record, float features[42]) {
    features[0]  = record->voltage;
    features[1]  = record->measured;
    features[2]  = record->min_val;
    features[3]  = record->max_val;
    features[4]  = record->std_dev;
    features[5]  = record->avg;
    features[6]  = record->median;
    features[7]  = record->bounce_back;
    features[8]  = record->drop;
    features[9]  = record->slope_bounce_back;
    features[10] = record->slope_drop;
    features[11] = record->min_volt_below_19;
    features[12] = record->max_volt_19_above;
    features[13] = record->start_voltage;
    features[14] = record->recovery_time_ms;
    features[15] = record->area_0_200ms;
    features[16] = record->count_below7;
    features[17] = record->count_below9;
    features[18] = record->count_below10;
    features[19] = record->curve_kurtosis;
    features[20] = record->curve_skew;
    features[21] = record->max_rise_rate_0_180;
    features[22] = record->max_fall_rate_0_180;
    features[23] = record->mean_abs_slope_0_180;
    features[24] = record->std_slope_0_180;
    features[25] = record->mean_abs_accel_0_180;
    features[26] = record->max_accel_0_180;
    features[27] = record->min_accel_0_180;
    features[28] = record->norm_energy_200ms;
    features[29] = record->rec_slope;
    features[30] = record->r_est;
    features[31] = record->spike_cnt;
    features[32] = record->dip_cnt;
    features[33] = record->prom_sum;
    features[34] = record->spike_w_mean_ms;
    features[35] = record->step_count_sust;
    features[36] = record->max_step_mag;
    features[37] = record->bp_low;
    features[38] = record->bp_mid;
    features[39] = record->bp_high;
    features[40] = record->bp_mid_ratio;
    features[41] = record->bp_high_ratio;
}

/* ===================== ADDED: tiny case-insensitive compare ===================== */
static int iequals(const char *a, const char *b) {
    for (; *a && *b; a++, b++) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
    }
    return *a == '\0' && *b == '\0';
}

/* ===================== ADDED: CSV line splitter (handles quotes) ===================== */
#define MAX_COLUMNS 512
static int split_csv_line(char *line, char **out_fields, int max_fields) {
    int count = 0;
    char *p = line;
    while (*p && count < max_fields) {
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '"') {
            p++; /* inside quotes */
            out_fields[count++] = p;
            while (*p) {
                if (*p == '"' && (*(p+1) == '"' )) { /* escaped quote "" -> " */
                    memmove(p, p+1, strlen(p)); /* collapse */
                    p++;
                    continue;
                }
                if (*p == '"' && (*(p+1) == ',' || *(p+1) == '\0' || *(p+1) == '\r' || *(p+1) == '\n')) {
                    break;
                }
                p++;
            }
            if (*p == '"') { *p = '\0'; p++; }
            if (*p == ',') { *p = '\0'; p++; }
        } else {
            out_fields[count++] = p;
            while (*p && *p != ',' && *p != '\r' && *p != '\n') p++;
            if (*p == ',') { *p = '\0'; p++; }
        }
        /* trim right */
        char *end = out_fields[count-1] + strlen(out_fields[count-1]);
        while (end > out_fields[count-1] && isspace((unsigned char)*(end-1))) { end--; }
        *end = '\0';
    }
    return count;
}

/* ===================== ADDED: dynamic line reader (no fixed row buffer) ===================== */
static ssize_t read_line_dynamic(FILE *f, char **buf, size_t *cap) {
    if (!*buf || *cap == 0) {
        *cap = 4096;
        *buf = (char*)malloc(*cap);
        if (!*buf) return -1;
    }
    size_t len = 0;
    for (;;) {
        if (!fgets(*buf + len, (int)(*cap - len), f)) {
            if (len == 0) return -1; /* EOF */
            break; /* partial line at EOF */
        }
        len += strlen(*buf + len);
        if (len > 0 && (*buf)[len-1] == '\n') break; /* full line */
        /* need more space */
        size_t newcap = (*cap < 262144 ? (*cap * 2) : (*cap + 262144));
        char *nb = (char*)realloc(*buf, newcap);
        if (!nb) return -1;
        *buf = nb; *cap = newcap;
    }
    return (ssize_t)len;
}

/* ===================== ADDED: output header and row writers ===================== */
static void write_output_header(FILE *fo) {
    fprintf(fo, "Test_Record_Detail_ID");
    const char *feat_names[42] = {
        "voltage","measured","min_val","max_val","std_dev","avg","median",
        "bounce_back","drop","slope_bounce_back","slope_drop",
        "min_volt_below_19","max_volt_19_above","start_voltage","recovery_time_ms",
        "area_0_200ms","count_below7","count_below9","count_below10","curve_kurtosis",
        "curve_skew","max_rise_rate_0_180","max_fall_rate_0_180","mean_abs_slope_0_180",
        "std_slope_0_180","mean_abs_accel_0_180","max_accel_0_180","min_accel_0_180",
        "norm_energy_200ms","rec_slope","r_est", "spike_cnt",
        "dip_cnt", "prom_sum", "spike_w_mean_ms", "step_count_sust", "max_step_mag",
        "bp_low", "bp_mid", "bp_high", "bp_mid_ratio", "bp_high_ratio"
    };
    for (int i=0;i<42;i++) fprintf(fo, ",%s", feat_names[i]);

    // MATCH THE ROW ORDER: Bad, then Good
    fprintf(fo, "\n");
}


static int is_blank_csv_line(const char *s) {
    if (!s) return 1;
    while (*s) {
        char c = *s++;
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n' && c != ',')
            return 0;
    }
    return 1;
}

static void write_row(FILE *fo, const char *id, const float features[42]) {
    fprintf(fo, "\"%s\"", id ? id : "");
    for (int i=0;i<42;i++) {
        if (isnan(features[i])) fprintf(fo, ",");
        else                    fprintf(fo, ",%.6f", features[i]);
    }
    // Match header: Good, Bad, Decision
    fprintf(fo, "\n");
}

static void trim_id_inplace(char *s) {
    if (!s) return;

    /* left trim: find first non-space */
    char *p = s;
    while (*p && isspace((unsigned char)*p)) p++;

    /* move to front if needed */
    if (p != s) memmove(s, p, strlen(p) + 1);

    /* right trim */
    size_t n = strlen(s);
    while (n && isspace((unsigned char)s[n-1])) s[--n] = '\0';

    /* strip one pair of wrapping quotes if present */
    n = strlen(s);
    if (n >= 2 && ((s[0] == '"' && s[n-1] == '"') || (s[0] == '\'' && s[n-1] == '\''))) {
        memmove(s, s+1, n-2);
        s[n-2] = '\0';
    }
}

/* ===== track emitted IDs to avoid duplicates ===== */
typedef struct { char **data; int count; int cap; } IdSet;

static void idset_init(IdSet *S){ S->data=NULL; S->count=0; S->cap=0; }

static int idset_contains(IdSet *S, const char *id){
    for (int i=0;i<S->count;i++) if (strcmp(S->data[i], id)==0) return 1;
    return 0;
}
static int idset_add(IdSet *S, const char *id){
    if (idset_contains(S, id)) return 0;
    if (S->count == S->cap){
        int ncap = S->cap ? S->cap*2 : 64;
        char **nb = (char**)realloc(S->data, ncap*sizeof(char*));
        if (!nb) return -1;
        S->data = nb; S->cap = ncap;
    }
    S->data[S->count] = strdup(id ? id : "");
    if (!S->data[S->count]) return -1;
    S->count++;
    return 1;
}
static void idset_free(IdSet *S){
    if (!S) return;
    for (int i=0;i<S->count;i++) free(S->data[i]);
    free(S->data);
}


/* ===================== ADDED: column index helper ===================== */
static int header_index(char **hdr, int n, const char **candidates, int n_candidates) {
    for (int i=0;i<n_candidates;i++) {
        for (int j=0;j<n;j++) {
            if (iequals(hdr[j], candidates[i])) return j;
        }
    }
    return -1;
}


/* ===================== CHANGED: main now loads CSV, runs inference, writes CSV ===================== */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_csv_path>\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];

    /* make a copy of input path so we can manipulate directory */
    char *path_copy = strdup(input_path);
    if (!path_copy) {
        fprintf(stderr, "Out of memory.\n");
        return 1;
    }
    char *dir = dirname(path_copy);  /* this modifies path_copy in place */

    /* construct output path in same directory */
    char out_path[4096];
    snprintf(out_path, sizeof(out_path), "%s/%s", dir, "MVT13_features_output.csv");
    free(path_copy);

    FILE *fi = fopen(input_path, "rb");
    if (!fi) {
        fprintf(stderr, "Failed to open input CSV: %s\n", input_path);
        return 1;
    }
    FILE *fo = fopen(out_path, "wb");
    if (!fo) {
        fclose(fi);
        fprintf(stderr, "Failed to open output CSV: %s\n", out_path);
        return 1;
    }

    /* dynamic buffers for lines */
    char *line = NULL;
    size_t cap = 0;
    ssize_t len;

    /* read header */
    len = read_line_dynamic(fi, &line, &cap);
    if (len < 0) {
        fprintf(stderr, "Empty input CSV.\n");
        fclose(fi); fclose(fo); free(line);
        return 1;
    }
    /* split header */
    char *header_line = strdup(line);
    if (!header_line) { fclose(fi); fclose(fo); free(line); return 1; }
    char *fields[MAX_COLUMNS];
    int nfields = split_csv_line(header_line, fields, MAX_COLUMNS);

    /* map required columns (aliases supported) */
    ColIdx col = {-1,-1,-1,-1,-1,-1};
    const char *c_id[]      = {"Test_Record_Detail_ID","test_record_detail_id","record_id","id"};
    const char *c_starter[] = {"StarterVoltageGraphpoints","starter_voltage_str","starter_voltage","starter_array"};
    const char *c_swver[]   = {"SoftwareVersionNumber","sw_version","softwareversion","version"};
    const char *c_volt[]    = {"Voltage","open_circuit_voltage","voc"};
    const char *c_meas[]    = {"Measured","measured_current","measured_cca","measured_value"};
    const char *c_ripple[]  = {"AlternatorRippleGraphpoints","alternator_ripple","ripple_array"};

    col.idx_id       = header_index(fields, nfields, c_id,      (int)(sizeof(c_id)/sizeof(c_id[0])));
    col.idx_starter  = header_index(fields, nfields, c_starter, (int)(sizeof(c_starter)/sizeof(c_starter[0])));
    col.idx_swver    = header_index(fields, nfields, c_swver,   (int)(sizeof(c_swver)/sizeof(c_swver[0])));
    col.idx_voltage  = header_index(fields, nfields, c_volt,    (int)(sizeof(c_volt)/sizeof(c_volt[0])));
    col.idx_measured = header_index(fields, nfields, c_meas,    (int)(sizeof(c_meas)/sizeof(c_meas[0])));
    col.idx_ripple   = header_index(fields, nfields, c_ripple,  (int)(sizeof(c_ripple)/sizeof(c_ripple[0])));

    free(header_line);

    if (col.idx_id < 0 || col.idx_starter < 0 || col.idx_swver < 0 ||
        col.idx_voltage < 0 || col.idx_measured < 0 || col.idx_ripple < 0) {
        fprintf(stderr, "Missing one or more required columns in CSV header.\n");
        fclose(fi); fclose(fo); free(line);
        return 1;
    }

    write_output_header(fo);

        /* process each data row */
    IdSet seen; idset_init(&seen);

    /* process each data row */
    while ((len = read_line_dynamic(fi, &line, &cap)) >= 0) {
        if (len == 0 || is_blank_csv_line(line)) continue;

        char *row = strdup(line);
        if (!row) { fprintf(stderr, "OOM on row dup.\n"); break; }

        char *cols[MAX_COLUMNS];
        int nf = split_csv_line(row, cols, MAX_COLUMNS);
        if (nf <= 0 || (nf == 1 && (!cols[0] || cols[0][0] == '\0'))) { free(row); continue; }

        /* extract fields */
        const char *id_raw   = (col.idx_id       < nf) ? cols[col.idx_id]       : "";
        const char *starter  = (col.idx_starter  < nf) ? cols[col.idx_starter]  : "";
        const char *swver    = (col.idx_swver    < nf) ? cols[col.idx_swver]    : "";
        const char *vstr     = (col.idx_voltage  < nf) ? cols[col.idx_voltage]  : "";
        const char *mstr     = (col.idx_measured < nf) ? cols[col.idx_measured] : "";
        const char *ripple   = (col.idx_ripple   < nf) ? cols[col.idx_ripple]   : "";

        /* normalize ID (trim spaces/quotes) into a scratch buffer */
        char idbuf[1024];
        snprintf(idbuf, sizeof(idbuf), "%s", id_raw ? id_raw : "");
        trim_id_inplace(idbuf);
        if (idbuf[0] == '\0') { printf("Dropping row %s: due to missing ID\n", line); free(row); continue; }  /* no ID → drop */

        float feats[42]; for (int i=0;i<42;i++) feats[i]=NAN;
        int invalid = 0;
        if (!invalid) {
            char *endp=NULL;
            float voltage = (float)strtof(vstr, &endp);
            if (endp == vstr) invalid = 1;

            long mtmp = strtol(mstr, &endp, 10);
            if (endp == mstr) invalid = 1;
            int16_t measured = (int16_t)mtmp;

            float ripple_vals[MAX_ALTERNATOR_RIPPLE];
            int ripple_count = process_alternator_ripple(ripple, ripple_vals);
            float ripple_sum = sum_alternator_ripple_array(ripple_vals, ripple_count);
            
            if (!invalid) {
                Record rec;
                preprocess_record(starter, swver, voltage, measured, &rec);

                /* inference: y[0]=Prob_Bad, y[1]=Prob_Good */
                pack_features(&rec, feats);
                /**
                float y[2] = {0.0f, 0.0f};
                score(feats, y);
                float p_good = y[1];
                float p_bad  = y[0];
                const char *decision = (p_good >= 0.5f) ? "GOOD_BATTERY" : "BAD_BATTERY";
                */
               
                /* mark ID as emitted only when we actually write the row */
                if (idset_add(&seen, idbuf) < 0) { free(row); break; }
                write_row(fo, idbuf, feats);
                 
                free(row);
                continue;
            }
        }
        /* invalid rows are dropped silently (no extra line in output) */
        free(row);
    }

    idset_free(&seen);


    free(line);
    fclose(fi);
    fclose(fo);
    return 0;
}
