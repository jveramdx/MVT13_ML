// test_numpy_functions.c
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include "numpy_functions.h"

#ifndef M_PIf
#define M_PIf 3.14159274101257324219f
#endif

// FFT tests removed. If FFT is added back, reintroduce tests here.

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
    return 0;
}
