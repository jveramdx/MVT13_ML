// numpy_functions.c  (reference-accurate, simple and readable)
#include "numpy_functions.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <complex.h>
#include <stdbool.h>

// Provide a fallback for M_PI on libcs that don't define it
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// ---------- small alloc helpers ----------
static void* xmalloc(size_t n){ void* p = malloc(n); if(!p){fprintf(stderr,"OOM\n"); exit(1);} return p; }
static void* xrealloc(void* p, size_t n){ void* q = realloc(p,n); if(!q){free(p); fprintf(stderr,"OOM\n"); exit(1);} return q; }

// ============================ FFT (simple DFT) ============================

fft_norm_t parse_fft_norm(const char* norm){
    if(!norm || strcmp(norm,"backward")==0) return FFT_NORM_BACKWARD;
    if(strcmp(norm,"forward")==0) return FFT_NORM_FORWARD;
    if(strcmp(norm,"ortho")==0)   return FFT_NORM_ORTHO;
    return FFT_NORM_BACKWARD;
}

float* np_fft_rfftfreq(int32_t n, float d){
    if(n <= 0 || d <= 0) return NULL;
    int32_t N = n/2 + 1;
    float step = 1.0f / ((float)n * d);
    float* f = (float*)xmalloc(sizeof(float) * (size_t)N);
    for(int32_t i=0;i<N;++i) f[i] = (float)i * step;
    return f;
}

float _Complex* np_fft_fft(const float _Complex* x, int32_t n, int32_t out_len,
                           fft_norm_t norm, void* scratch){
    (void)scratch; // unused
    if(n <= 0 || out_len <= 0) return NULL;

    float _Complex* X = (float _Complex*)xmalloc(sizeof(float _Complex) * (size_t)out_len);
    const float twoPiN = 2.0f*(float)M_PI / (float)n;

    // Reference DFT; O(n^2) but deterministic and dependency-free
    for(int32_t k=0; k<out_len; ++k){
        float _Complex acc = 0.0f + 0.0f*I;
        for(int32_t t=0; t<n; ++t){
            float ang = -twoPiN * (float)(k % n) * (float)t; // wrap k to n
            float c = cosf(ang), s = sinf(ang);
            acc += x[t] * (c + s*I);
        }
        X[k] = acc;
    }

    if(norm == FFT_NORM_FORWARD){
        float s = 1.0f/(float)n; for(int32_t k=0;k<out_len;++k) X[k] *= s;
    }else if(norm == FFT_NORM_ORTHO){
        float s = 1.0f/sqrtf((float)n); for(int32_t k=0;k<out_len;++k) X[k] *= s;
    }
    // BACKWARD: no scaling
    return X;
}

// ============================ find_peaks & medfilt (reference impls) ============================

// (Implementations adapted from your prior file; unchanged API)
typedef struct { bool has_min, has_max; float vmin, vmax; } interval_t;
static inline int32_t imin32(int32_t a,int32_t b){ return a<b?a:b; }
static inline int32_t imax32(int32_t a,int32_t b){ return a>b?a:b; }
static inline float  fmax32(float a,float b){ return a>b?a:b; }
static inline bool in_interval_f(float x, interval_t iv){
    if(iv.has_min && x < iv.vmin) return false;
    if(iv.has_max && x > iv.vmax) return false;
    return true;
}

static void local_maxima_with_plateaus_f(const float* x, int32_t n,
                                         int32_t** out_peaks, int32_t* out_n,
                                         int32_t** out_L, int32_t** out_R){
    int32_t cap = imax32(8, n/4 + 1);
    int32_t* peaks=(int32_t*)xmalloc(sizeof(int32_t)*(size_t)cap);
    int32_t* L    =(int32_t*)xmalloc(sizeof(int32_t)*(size_t)cap);
    int32_t* R    =(int32_t*)xmalloc(sizeof(int32_t)*(size_t)cap);
    int32_t k=0, i=1;
    while(i < n-1){
        if(x[i] > x[i-1] && x[i] > x[i+1]){
            if(k==cap){ cap*=2; peaks=xrealloc(peaks,sizeof(int32_t)*(size_t)cap);
                        L=xrealloc(L,sizeof(int32_t)*(size_t)cap);
                        R=xrealloc(R,sizeof(int32_t)*(size_t)cap); }
            peaks[k]=i; L[k]=i; R[k]=i; k++; i++; continue;
        }
        if(x[i] >= x[i-1] && x[i] == x[i+1]){
            int32_t s=i, e=i+1;
            while(e<n-1 && x[e]==x[e+1]) e++;
            float left=x[s-1], right=x[e+1], level=x[i];
            if(level>left && level>right){
                int32_t mid=(s+e)/2;
                if(k==cap){ cap*=2; peaks=xrealloc(peaks,sizeof(int32_t)*(size_t)cap);
                            L=xrealloc(L,sizeof(int32_t)*(size_t)cap);
                            R=xrealloc(R,sizeof(int32_t)*(size_t)cap); }
                peaks[k]=mid; L[k]=s; R[k]=e; k++;
            }
            i=e+1; continue;
        }
        i++;
    }
    *out_peaks=peaks; *out_n=k; *out_L=L; *out_R=R;
}

static void compute_thresholds_f(const float* x, const int32_t* peaks, int32_t n_peaks,
                                 float* left_thr, float* right_thr){
    for(int32_t i=0;i<n_peaks;++i){
        int32_t p=peaks[i];
        left_thr[i]  = x[p] - x[p-1];
        right_thr[i] = x[p] - x[p+1];
    }
}

static int32_t* select_by_distance_f(const int32_t* peaks, int32_t n_peaks,
                                     const float* x, int32_t distance, int32_t* out_n){
    // order peaks by height (desc), then greedy distance
    int32_t* order=(int32_t*)xmalloc(sizeof(int32_t)*(size_t)n_peaks);
    for(int32_t i=0;i<n_peaks;++i) order[i]=i;
    for(int32_t i=0;i<n_peaks;++i){
        int32_t best=i;
        for(int32_t j=i+1;j<n_peaks;++j)
            if(x[peaks[order[j]]] > x[peaks[order[best]]]) best=j;
        int32_t t=order[i]; order[i]=order[best]; order[best]=t;
    }
    bool* taken=(bool*)xmalloc(sizeof(bool)*(size_t)n_peaks); memset(taken,0,(size_t)n_peaks);
    int32_t* keep=(int32_t*)xmalloc(sizeof(int32_t)*(size_t)n_peaks); int32_t k=0;
    for(int32_t oi=0;oi<n_peaks;++oi){
        int32_t idx=order[oi]; if(taken[idx]) continue;
        int32_t p=peaks[idx]; keep[k++]=idx;
        for(int32_t j=0;j<n_peaks;++j)
            if(!taken[j] && j!=idx && abs(peaks[j]-p) < distance) taken[j]=true;
    }
    free(order); free(taken);

    int32_t* sel=(int32_t*)xmalloc(sizeof(int32_t)*(size_t)k);
    for(int32_t i=0;i<k;++i) sel[i]=peaks[keep[i]];
    for(int32_t i=0;i<k;++i) for(int32_t j=i+1;j<k;++j) if(sel[j]<sel[i]){int32_t t=sel[i]; sel[i]=sel[j]; sel[j]=t;}
    *out_n=k; free(keep); return sel;
}

static void compute_prominence_f(const float* x, int32_t n,
                                 const int32_t* peaks, int32_t n_peaks,
                                 int32_t wlen,
                                 float* prom, int32_t* lb, int32_t* rb){
    for(int32_t i=0;i<n_peaks;++i){
        int32_t p=peaks[i]; float peak_h=x[p];
        int32_t L=0,R=n-1;
        if(wlen>0){ int32_t half=wlen/2; L = (p-half>0) ? (p-half) : 0; R = (p+half<n-1) ? (p+half) : (n-1); }
        float minL=peak_h, minR=peak_h; int32_t lbi=L, rbi=R; float cur=peak_h;
        for(int32_t j=p; j>=L; --j){ if(x[j]>peak_h) break; if(x[j]<cur){ cur=x[j]; lbi=j; } } minL=cur;
        cur=peak_h;
        for(int32_t j=p; j<=R; ++j){ if(x[j]>peak_h) break; if(x[j]<cur){ cur=x[j]; rbi=j; } } minR=cur;
        float ref = (minL > minR) ? minL : minR;
        prom[i] = peak_h - ref; lb[i]=lbi; rb[i]=rbi;
    }
}

static void compute_widths_f(const float* x, const int32_t* peaks, int32_t n_peaks,
                             const float* prom, const int32_t* lb, const int32_t* rb,
                             float rel_height,
                             float* widths, float* wh, float* lips, float* rips){
    for(int32_t i=0;i<n_peaks;++i){
        int32_t p=peaks[i]; float h = x[p] - prom[i]*rel_height; wh[i]=h;

        float li=(float)p; bool fl=false;
        for(int32_t j=p; j>lb[i]; --j){
            if(x[j-1] <= h && x[j] >= h){
                float y1=x[j-1], y2=x[j]; float t=(h - y1) / (y2 - y1 + 1e-30f);
                li = (float)(j-1) + t; fl=true; break;
            }
        }
        if(!fl) li=(float)lb[i];

        float ri=(float)p; bool fr=false;
        for(int32_t j=p; j<rb[i]; ++j){
            if(x[j+1] <= h && x[j] >= h){
                float y1=x[j], y2=x[j+1]; float t=(h - y1) / (y2 - y1 + 1e-30f);
                ri = (float)j + t; fr=true; break;
            }
        }
        if(!fr) ri=(float)rb[i];

        lips[i]=li; rips[i]=ri; widths[i]=ri-li;
    }
}

peak_result_t find_peaks_ref(const float* x, int32_t n,
                             const float height[2],
                             const float threshold[2],
                             int32_t distance,
                             const float prominence[2],
                             const float width[2],
                             int32_t wlen,
                             float rel_height,
                             const float plateau_size[2]){
    peak_result_t res; memset(&res, 0, sizeof(res));
    if(n < 3) return res;
    if(distance < 0){ fprintf(stderr,"distance must be >= 0\n"); return res; }

    interval_t Ih={0}, It={0}, Ip={0}, Iw={0}, Ipl={0};
    if(height){    if(!isnan(height[0])){ Ih.has_min=true; Ih.vmin=height[0]; }
                   if(!isnan(height[1])){ Ih.has_max=true; Ih.vmax=height[1]; } }
    if(threshold){ if(!isnan(threshold[0])){ It.has_min=true; It.vmin=threshold[0]; }
                   if(!isnan(threshold[1])){ It.has_max=true; It.vmax=threshold[1]; } }
    if(prominence){if(!isnan(prominence[0])){ Ip.has_min=true; Ip.vmin=prominence[0]; }
                   if(!isnan(prominence[1])){ Ip.has_max=true; Ip.vmax=prominence[1]; } }
    if(width){     if(!isnan(width[0])){ Iw.has_min=true; Iw.vmin=width[0]; }
                   if(!isnan(width[1])){ Iw.has_max=true; Iw.vmax=width[1]; } }
    if(plateau_size){ if(!isnan(plateau_size[0])){ Ipl.has_min=true; Ipl.vmin=plateau_size[0]; }
                      if(!isnan(plateau_size[1])){ Ipl.has_max=true; Ipl.vmax=plateau_size[1]; } }

    int32_t *P=NULL,*L=NULL,*R=NULL, K=0;
    local_maxima_with_plateaus_f(x,n,&P,&K,&L,&R);
    if(K==0){ free(P); free(L); free(R); return res; }

    // plateau filter
    int32_t* keep=(int32_t*)xmalloc(sizeof(int32_t)*(size_t)K); int32_t kn=0;
    int32_t* plats=NULL;
    if(Ipl.has_min || Ipl.has_max){
        plats=(int32_t*)xmalloc(sizeof(int32_t)*(size_t)K);
        for(int32_t i=0;i<K;++i) plats[i]=R[i]-L[i]+1;
        for(int32_t i=0;i<K;++i) if(in_interval_f((float)plats[i], Ipl)) keep[kn++]=i;
    } else { for(int32_t i=0;i<K;++i) keep[kn++]=i; }

    int32_t KK=kn;
    int32_t* PP=(int32_t*)xmalloc(sizeof(int32_t)*(size_t)KK);
    int32_t* LL=(int32_t*)xmalloc(sizeof(int32_t)*(size_t)KK);
    int32_t* RR=(int32_t*)xmalloc(sizeof(int32_t)*(size_t)KK);
    for(int32_t i=0;i<KK;++i){ PP[i]=P[keep[i]]; LL[i]=L[keep[i]]; RR[i]=R[keep[i]]; }
    free(P); free(L); free(R);

    float* PH=NULL;
    if(Ih.has_min || Ih.has_max){
        PH=(float*)xmalloc(sizeof(float)*(size_t)KK);
        int32_t w=0;
        for(int32_t i=0;i<KK;++i){
            PH[i]=x[PP[i]];
            if(in_interval_f(PH[i], Ih)){ PP[w]=PP[i]; LL[w]=LL[i]; RR[w]=RR[i]; PH[w]=PH[i]; w++; }
        }
        KK=w;
    }

    float *LT=NULL,*RT=NULL;
    if(It.has_min || It.has_max){
        LT=(float*)xmalloc(sizeof(float)*(size_t)KK);
        RT=(float*)xmalloc(sizeof(float)*(size_t)KK);
        compute_thresholds_f(x,PP,KK,LT,RT);
        int32_t w=0;
        for(int32_t i=0;i<KK;++i){
            float lt=LT[i], rt=RT[i];
            if(in_interval_f(lt,It) && in_interval_f(rt,It)){
                PP[w]=PP[i]; LL[w]=LL[i]; RR[w]=RR[i];
                if(PH) PH[w]=PH[i]; LT[w]=lt; RT[w]=rt; w++;
            }
        }
        KK=w;
    }

    if(distance >= 1 && KK>1){
        int32_t newN=0; int32_t* sel = select_by_distance_f(PP,KK,x,distance,&newN);
        int32_t* nLL=(int32_t*)xmalloc(sizeof(int32_t)*(size_t)newN);
        int32_t* nRR=(int32_t*)xmalloc(sizeof(int32_t)*(size_t)newN);
        float* nPH = PH? (float*)xmalloc(sizeof(float)*(size_t)newN) : NULL;
        float* nLT = LT? (float*)xmalloc(sizeof(float)*(size_t)newN) : NULL;
        float* nRT = RT? (float*)xmalloc(sizeof(float)*(size_t)newN) : NULL;

        for(int32_t i=0;i<newN;++i){
            int32_t pos=sel[i]; int32_t j=0; for(; j<KK; ++j) if(PP[j]==pos) break;
            nLL[i]=LL[j]; nRR[i]=RR[j];
            if(nPH) nPH[i]=x[pos];
            if(nLT) nLT[i]=LT[j];
            if(nRT) nRT[i]=RT[j];
        }
        free(PP); free(LL); free(RR);
        if(PH){ free(PH); PH=nPH; }
        if(LT){ free(LT); LT=nLT; }
        if(RT){ free(RT); RT=nRT; }
        PP=sel; KK=newN; LL=nLL; RR=nRR;
    }

    float *PR=NULL; int32_t *LB=NULL,*RB=NULL;
    if((prominence && (!isnan(prominence[0]) || !isnan(prominence[1]))) ||
       (width && (!isnan(width[0]) || !isnan(width[1])))){
        PR =(float*)xmalloc(sizeof(float)*(size_t)KK);
        LB =(int32_t*)xmalloc(sizeof(int32_t)*(size_t)KK);
        RB =(int32_t*)xmalloc(sizeof(int32_t)*(size_t)KK);
        compute_prominence_f(x,n,PP,KK,wlen,PR,LB,RB);

        if(prominence && (!isnan(prominence[0]) || !isnan(prominence[1]))){
            interval_t Ip={0};
            if(!isnan(prominence[0])){ Ip.has_min=true; Ip.vmin=prominence[0]; }
            if(!isnan(prominence[1])){ Ip.has_max=true; Ip.vmax=prominence[1]; }
            int32_t w=0;
            for(int32_t i=0;i<KK;++i){
                if(in_interval_f(PR[i],Ip)){
                    PP[w]=PP[i]; LL[w]=LL[i]; RR[w]=RR[i]; PR[w]=PR[i]; LB[w]=LB[i]; RB[w]=RB[i];
                    if(PH) PH[w]=PH[i]; if(LT){ LT[w]=LT[i]; RT[w]=RT[i]; } w++;
                }
            }
            KK=w;
        }
    }

    float *W=NULL,*WH=NULL,*LIPS=NULL,*RIPS=NULL;
    if(width && (!isnan(width[0]) || !isnan(width[1]))){
        W=(float*)xmalloc(sizeof(float)*(size_t)KK);
        WH=(float*)xmalloc(sizeof(float)*(size_t)KK);
        LIPS=(float*)xmalloc(sizeof(float)*(size_t)KK);
        RIPS=(float*)xmalloc(sizeof(float)*(size_t)KK);
        compute_widths_f(x,PP,KK,PR,LB,RB,rel_height,W,WH,LIPS,RIPS);

        interval_t Iw={0};
        if(!isnan(width[0])){ Iw.has_min=true; Iw.vmin=width[0]; }
        if(!isnan(width[1])){ Iw.has_max=true; Iw.vmax=width[1]; }
        int32_t w=0;
        for(int32_t i=0;i<KK;++i){
            if(in_interval_f(W[i],Iw)){
                PP[w]=PP[i]; LL[w]=LL[i]; RR[w]=RR[i];
                W[w]=W[i]; WH[w]=WH[i]; LIPS[w]=LIPS[i]; RIPS[w]=RIPS[i];
                if(PH) PH[w]=PH[i]; if(LT){ LT[w]=LT[i]; RT[w]=RT[i]; }
                if(PR){ PR[w]=PR[i]; LB[w]=LB[i]; RB[w]=RB[i]; }
                w++;
            }
        }
        KK=w;
    }

    res.n_peaks = KK;
    res.indices = (int32_t*)xmalloc(sizeof(int32_t)*(size_t)KK);
    for(int32_t i=0;i<KK;++i) res.indices[i]=PP[i];

    res.peak_heights = PH;
    res.left_thresholds = LT; res.right_thresholds = RT;
    res.prominences = PR; res.left_bases = LB; res.right_bases = RB;
    res.widths = W; res.width_heights = WH; res.left_ips = LIPS; res.right_ips = RIPS;

    if(Ipl.has_min || Ipl.has_max){
        int32_t* left_edges=(int32_t*)xmalloc(sizeof(int32_t)*(size_t)KK);
        int32_t* right_edges=(int32_t*)xmalloc(sizeof(int32_t)*(size_t)KK);
        int32_t* plats_out=(int32_t*)xmalloc(sizeof(int32_t)*(size_t)KK);
        for(int32_t i=0;i<KK;++i){ left_edges[i]=LL[i]; right_edges[i]=RR[i]; plats_out[i]=RR[i]-LL[i]+1; }
        res.left_edges = left_edges; res.right_edges = right_edges; res.plateau_sizes = plats_out;
        if(plats) free(plats);
    }

    free(PP); free(LL); free(RR); free(keep);
    return res;
}

void free_peak_result(peak_result_t* r){
    if(!r) return;
    free(r->indices);
    free(r->peak_heights);
    free(r->left_thresholds); free(r->right_thresholds);
    free(r->prominences); free(r->left_bases); free(r->right_bases);
    free(r->widths); free(r->width_heights); free(r->left_ips); free(r->right_ips);
    free(r->left_edges); free(r->right_edges); free(r->plateau_sizes);
    memset(r, 0, sizeof(*r));
}

// ============================ medfilt (N-D) ============================

static uint32_t prod_u32(const int32_t* a, int32_t n){
    uint64_t p=1; for(int32_t i=0;i<n;++i) p *= (uint64_t)(uint32_t)a[i];
    if(p > 0xFFFFFFFFu){ fprintf(stderr,"medfilt_nd: product overflow\n"); exit(1); }
    return (uint32_t)p;
}
static void unravel_index_u32(uint32_t idx, const int32_t* dims, int32_t nd, int32_t* coords){
    for(int32_t d=nd-1; d>=0; --d){ int32_t s=dims[d]; coords[d]=(int32_t)(idx % (uint32_t)s); idx/=(uint32_t)s; }
}
static uint32_t ravel_index_u32(const int32_t* coords, const int32_t* dims, int32_t nd){
    uint64_t idx=0;
    for(int32_t d=0; d<nd; ++d){
        idx = idx*(uint64_t)(uint32_t)dims[d] + (uint64_t)(uint32_t)coords[d];
        if(idx > 0xFFFFFFFFu){ fprintf(stderr,"medfilt_nd: index overflow\n"); exit(1); }
    }
    return (uint32_t)idx;
}
static int cmp_f32_qsort(const void* a, const void* b){
    float da=*(const float*)a, db=*(const float*)b; return (da>db)-(da<db);
}

void medfilt_nd(const float* volume, const int32_t* dims, int32_t nd,
                const int32_t* kernel, float* out){
    for(int32_t d=0; d<nd; ++d){
        if(kernel[d]%2 != 1){ fprintf(stderr,"kernel sizes must be odd\n"); return; }
        if(dims[d] <= 0){ fprintf(stderr,"invalid dims\n"); return; }
    }
    uint32_t total = prod_u32(dims, nd);

    int32_t* coords=(int32_t*)xmalloc(sizeof(int32_t)*(size_t)nd);
    int32_t* half  =(int32_t*)xmalloc(sizeof(int32_t)*(size_t)nd);
    for(int32_t d=0; d<nd; ++d) half[d]=kernel[d]/2;

    uint64_t win_sz64=1; for(int32_t d=0; d<nd; ++d) win_sz64 *= (uint64_t)(uint32_t)kernel[d];
    if(win_sz64 > 0xFFFFFFFFu){ fprintf(stderr,"kernel window too large\n"); exit(1); }
    uint32_t win_size=(uint32_t)win_sz64;

    float* window=(float*)xmalloc(sizeof(float)*(size_t)win_size);
    int32_t* kcoord=(int32_t*)xmalloc(sizeof(int32_t)*(size_t)nd);
    int32_t* pos   =(int32_t*)xmalloc(sizeof(int32_t)*(size_t)nd);

    for(uint32_t idx=0; idx<total; ++idx){
        unravel_index_u32(idx, dims, nd, coords);
        uint32_t w=0;
        for(uint32_t wi=0; wi<win_size; ++wi){
            uint32_t t=wi;
            for(int32_t d=nd-1; d>=0; --d){ int32_t span=kernel[d]; kcoord[d]=(int32_t)(t%(uint32_t)span)-half[d]; t/=(uint32_t)span; }
            bool inb=true;
            for(int32_t d=0; d<nd; ++d){
                int32_t cd=coords[d]+kcoord[d]; pos[d]=cd;
                if(cd<0 || cd>=dims[d]){ inb=false; break; }
            }
            window[w++] = inb ? volume[ravel_index_u32(pos, dims, nd)] : 0.0f;
        }
        qsort(window, (size_t)w, sizeof(float), cmp_f32_qsort);
        float med = ( (w & 1u) ? window[w/2] : 0.5f*(window[w/2 - 1] + window[w/2]) );
        out[idx] = med;
    }
    free(window); free(kcoord); free(pos); free(coords); free(half);
}

// ============================ stats / nan-aware ============================

float mean_f32(const float* a, int32_t n){
    float s=0.0f; for(int32_t i=0;i<n;++i) s+=a[i]; return s/(float)n;
}

float nanmin(const float* arr, int n){
    float minv=FLT_MAX; int seen=0;
    for(int i=0;i<n;i++){ float v=arr[i]; if(!isnan(v)){ if(!seen || v<minv){minv=v;} seen=1; } }
    return seen ? minv : NAN;
}

float nanmax(const float* arr, int n){
    float maxv=-FLT_MAX; int seen=0;
    for(int i=0;i<n;i++){ float v=arr[i]; if(!isnan(v)){ if(!seen || v>maxv){maxv=v;} seen=1; } }
    return seen ? maxv : NAN;
}

float nanmean(const float* arr, int n){
    float sum=0.0f; int cnt=0;
    for(int i=0;i<n;i++){ float v=arr[i]; if(!isnan(v)){ sum+=v; cnt++; } }
    return cnt>0 ? sum/(float)cnt : NAN;
}

float nanstd(const float* arr, int n){
    float m=nanmean(arr,n); if(isnan(m)) return NAN;
    float ss=0.0f; int cnt=0;
    for(int i=0;i<n;i++){ float v=arr[i]; if(!isnan(v)){ float d=v-m; ss+=d*d; cnt++; } }
    return cnt>0 ? sqrtf(ss/(float)cnt) : NAN;
}

float nanmedian(const float* arr, int n){
    int cnt=0; for(int i=0;i<n;i++) if(!isnan(arr[i])) cnt++;
    if(cnt==0) return NAN;
    float* tmp=(float*)xmalloc(sizeof(float)*(size_t)cnt);
    int k=0; for(int i=0;i<n;i++) if(!isnan(arr[i])) tmp[k++]=arr[i];
    qsort(tmp,(size_t)cnt,sizeof(float),cmp_f32_qsort);
    float med = (cnt & 1) ? tmp[cnt/2] : 0.5f*(tmp[cnt/2-1]+tmp[cnt/2]);
    free(tmp); return med;
}

float nan_kurtosis(const float* a, int n){
    // unbiased Pearson (Fisher=False, bias=False)
    int cnt=0; float mu=0.0f;
    for(int i=0;i<n;i++){ float v=a[i]; if(!isnan(v)){ mu+=v; cnt++; } }
    if(cnt<4) return NAN;
    mu /= (float)cnt;
    float s2=0.0f, s4=0.0f;
    for(int i=0;i<n;i++){ float v=a[i]; if(!isnan(v)){ float d=v-mu; float d2=d*d; s2+=d2; s4+=d2*d2; } }
    float m2=s2/(float)cnt, m4=s4/(float)cnt;
    if(m2 <= DBL_EPSILON*mu*mu) return NAN;
    float g2 = m4/(m2*m2);
    float nn=(float)cnt;
    float num=(nn*nn - 1.0f)*g2 - 3.0f*(nn-1.0f)*(nn-1.0f);
    float den=(nn-2.0f)*(nn-3.0f);
    if(den==0.0f) return NAN;
    return num/den + 3.0f; // Pearson
}

float nan_skew(const float* a, int n){
    int cnt=0; float mu=0.0f;
    for(int i=0;i<n;i++){ float v=a[i]; if(!isnan(v)){ mu+=v; cnt++; } }
    if(cnt<3) return NAN;
    mu/=(float)cnt;
    float s2=0.0f, s3=0.0f;
    for(int i=0;i<n;i++){ float v=a[i]; if(!isnan(v)){ float d=v-mu; float d2=d*d; s2+=d2; s3+=d2*d; } }
    float m2=s2/(float)cnt; if(m2 <= DBL_EPSILON*mu*mu) return NAN;
    float g1 = s3/(float)cnt / powf(m2,1.5f);
    return sqrtf((float)cnt*((float)cnt-1.0f))/((float)cnt-2.0f) * g1;
}

// ============================ tiny numpy-like utils ============================

int32_t nanargmin_first_n(const float* x, int32_t n){
    int32_t idx=-1; float mv=FLT_MAX;
    for(int32_t i=0;i<n;++i){ float v=x[i]; if(!isnan(v) && v<mv){ mv=v; idx=i; } }
    return idx;
}

int32_t first_nan_after_i0_f32(const float* x, int32_t n, int32_t i0){
    for(int32_t j=i0; j<n; ++j) if(isnan(x[j])) return j;
    return -1;
}
