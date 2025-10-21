// test_numpy_functions.c
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include "numpy_functions.h"

#ifndef M_PIf
#define M_PIf 3.14159274101257324219f
#endif

// This tests if the numpy like functions are running or working. We will validate
// against numpy outputs in Python separately since floating point results may vary slightly.


static void test_rfftfreq() {
    for(int t=0;t<10;++t){
        int n = 8 + 2*t;
        float d = 0.1f + 0.05f*t;
        float* f = np_fft_rfftfreq(n, d);
        int N = n/2+1;
        printf("rfftfreq,%d,", t);
        for(int i=0;i<N;++i){
            printf("%g", f[i]);
            if(i<N-1) printf(";");
        }
        printf("\n");
        free(f);
    }
}

static void test_find_peaks() {
    for(int t=0;t<10;++t){
        int n = 20;
        float* x = malloc(n*sizeof(float));
        for(int i=0;i<n;++i){
            x[i] = sinf(0.3f*i) + (t%3==0?0.5f:0.0f); // vary shape
        }
        float NaN = NAN;
        float H[2] = {NaN, NaN};
        if(t%2==0) H[0] = 0.2f; // add height condition sometimes
        peak_result_t pr = find_peaks_ref(x, n, H, NULL, (t%3)+1,
                                          NULL, NULL, 0, 0.5f, NULL);
        printf("find_peaks,%d,", t);
        for(int i=0;i<pr.n_peaks;++i){
            printf("%d", pr.indices[i]);
            if(i<pr.n_peaks-1) printf(";");
        }
        printf("\n");
        free_peak_result(&pr);
        free(x);
    }
}

static void test_medfilt() {
    for(int t=0;t<10;++t){
        int dims[1] = {9};
        int kernel[1] = { (t%3)*2+3 }; // 3,5,7
        float vol[9];
        for(int i=0;i<9;++i) vol[i] = (float)((i+t)%5);
        float out[9];
        medfilt_nd(vol,dims,1,kernel,out);
        printf("medfilt,%d,", t);
        for(int i=0;i<9;++i){
            printf("%g", out[i]);
            if(i<8) printf(";");
        }
        printf("\n");
    }
}

int main() {
    // FFT tests removed because FFT implementation is intentionally omitted.
    test_rfftfreq();
    test_find_peaks();
    test_medfilt();
    /* New: test feature helper functions implemented in numpy_functions.c */
    {
        int n = 32;
        float x[32];
        for (int i = 0; i < n; ++i) x[i] = sinf(0.2f * i) + ((i%7)==0 ? 1.0f : 0.0f);
        /* create a residual-like signal (trend removed) */
        float resid[32];
        for (int i=0;i<n;++i) resid[i] = x[i] - 0.5f * cosf(0.15f*i);

        printf("feature_test,spike_count,%d\n", compute_spike_count(resid, n));
        printf("feature_test,dip_count,%d\n", compute_dip_count(resid, n));
        printf("feature_test,prom_sum,%g\n", compute_spike_prom_sum(resid, n));
        printf("feature_test,spike_width_mean_ms,%g\n", compute_spike_width_mean_ms(resid, n, 10.0f));
        printf("feature_test,longest_flat,%d\n", compute_longest_flat(x, n, 1e-3f));
        printf("feature_test,hf_energy,%g\n", compute_hf_energy(x, n));
        printf("feature_test,roll_var,%g\n", compute_roll_var(x, n, 5));
        printf("feature_test,edge_start_diff,%g\n", compute_edge_start_diff(x, n, 5));
        printf("feature_test,edge_end_diff,%g\n", compute_edge_end_diff(x, n, 5));
        printf("feature_test,min_drop,%g\n", compute_min_drop(x, n));
        printf("feature_test,recovery_slope,%g\n", compute_recovery_slope(x, n, 0.5f, 10.0f));
        printf("feature_test,poly_resid,%g\n", compute_poly_resid(x, n, 1));
        printf("feature_test,segment_slope_var,%g\n", compute_segment_slope_var(x, n, 8));
        printf("feature_test,zero_cross_rate,%g\n", compute_zero_cross_rate(x, n));
        printf("feature_test,step_count_sust,%d\n", compute_step_count_sustained(x, n, 4));
        printf("feature_test,max_step_mag,%g\n", compute_max_step_mag(x, n, 4));
        printf("feature_test,rel_below_frac,%g\n", compute_rel_below_frac(x, n, 0.0f));
        printf("feature_test,rel_below_longest_ms,%d\n", compute_rel_below_longest_ms(x, n, 0.0f, 10.0f));
        printf("feature_test,win_range_max,%g\n", compute_win_range_max(x, n, 10));
        printf("feature_test,tail_std,%g\n", compute_tail_std(x, n, 5));
        printf("feature_test,tail_ac1,%g\n", compute_tail_ac1(x, n, 5));
        printf("feature_test,crest_factor,%g\n", compute_crest_factor(x, n));
        printf("feature_test,line_length,%g\n", compute_line_length(x, n));
        printf("feature_test,mid_duty_cycle_low,%g\n", compute_mid_duty_cycle_low(x, n, 0.0f));
    }
    return 0;
}
