#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <libgen.h>
#include "numpy_functions.h"
#include <stdbool.h>
#include "record.h"
#include "features.h"

#define MAX_POINTS 508
#define MAX_ALTERNATOR_RIPPLE 257

typedef struct {
    int idx_id;
    int idx_starter;
    int idx_swver;
    int idx_voltage;
    int idx_measured;
    int idx_ripple;
} ColIdx;

// ------------ small helpers ------------
static int hex_to_signed_int(const char *h) {
    if (!h) return 0;
    unsigned int val = 0;
    sscanf(h, "%8x", &val);
    if (val & 0x80000000U) return (int)(val - 0x100000000U);
    return (int)val;
}

static float convert_value_version(const char *token) {
    if (!token || token[0]=='\0') return NAN;
    char *endptr;
    long val = strtol(token, &endptr, 16);
    if (endptr == token) return NAN;
    return (float)val / 100.0f;
}

static int16_t process_starter_voltage_version(const char *input, float values[MAX_POINTS]) {
    int16_t written = 0, last_non_nan = -1;
    for (int i = 0; i < MAX_POINTS; i++) values[i] = NAN;
    if (!input) return 0;
    char *copy = strdup(input); if (!copy) return 0;
    char *save = NULL, *token = strtok_r(copy, ":", &save);
    while (token && written < MAX_POINTS) {
        float v = convert_value_version(token);
        values[written] = v;
        if (!isnan(v)) last_non_nan = written;
        written++;
        token = strtok_r(NULL, ":", &save);
    }
    free(copy);
    return (int16_t)((last_non_nan >= 0) ? (last_non_nan + 1) : 0);
}

static int process_alternator_ripple(const char *input, float values[MAX_ALTERNATOR_RIPPLE]){
    int16_t count = 0;
    if(!input) return 0;
    char *copy = strdup(input); if (!copy) return 0;
    for(char *tok=strtok(copy,":"); tok && count<MAX_ALTERNATOR_RIPPLE; tok=strtok(NULL,":"))
        values[count++] = (float)hex_to_signed_int(tok);
    free(copy);
    return count;
}

static float sum_alternator_ripple_array(const float *arr, int n) {
    float sum = 0.0f; for (int i = 0; i < n; i++) sum += arr[i]; return sum;
}

static void preprocess_record(const char *starter_str, const char *software_version_str,
                              float voltage, float measured, Record *out) {
    (void)software_version_str; // currently unused
    float fvalues[MAX_POINTS]; for (int i = 0; i < MAX_POINTS; i++) fvalues[i] = NAN;
    int16_t n = process_starter_voltage_version(starter_str, fvalues);
    compute_features(fvalues, n, out, voltage, (int16_t)measured);
}

// -------- feature packing (42) --------
static void pack_features(const Record *r, float f[42]) {
    f[0]=r->voltage; f[1]=r->measured; f[2]=r->min_val; f[3]=r->max_val; f[4]=r->std_dev;
    f[5]=r->avg; f[6]=r->median; f[7]=r->bounce_back; f[8]=r->drop; f[9]=r->slope_bounce_back;
    f[10]=r->slope_drop; f[11]=r->min_volt_below_19; f[12]=r->max_volt_19_above; f[13]=r->start_voltage;
    f[14]=r->recovery_time_ms; f[15]=r->area_0_200ms; f[16]=r->count_below7; f[17]=r->count_below9;
    f[18]=r->count_below10; f[19]=r->curve_kurtosis; f[20]=r->curve_skew; f[21]=r->max_rise_rate_0_180;
    f[22]=r->max_fall_rate_0_180; f[23]=r->mean_abs_slope_0_180; f[24]=r->std_slope_0_180;
    f[25]=r->mean_abs_accel_0_180; f[26]=r->max_accel_0_180; f[27]=r->min_accel_0_180;
    f[28]=r->norm_energy_200ms; f[29]=r->rec_slope; f[30]=r->r_est; f[31]=r->spike_cnt;
    f[32]=r->dip_cnt; f[33]=r->prom_sum; f[34]=r->spike_w_mean_ms; f[35]=r->step_count_sust;
    f[36]=r->max_step_mag; f[37]=r->bp_low; f[38]=r->bp_mid; f[39]=r->bp_high;
    f[40]=r->bp_mid_ratio; f[41]=r->bp_high_ratio;
}

static int iequals(const char *a, const char *b) {
    for (; *a && *b; a++, b++) if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
    return *a == '\0' && *b == '\0';
}

#define MAX_COLUMNS 512
static int split_csv_line(char *line, char **out_fields, int max_fields) {
    int count = 0; char *p = line;
    while (*p && count < max_fields) {
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '"') {
            p++; out_fields[count++] = p;
            while (*p) {
                if (*p=='"' && (*(p+1)=='"')) { memmove(p, p+1, strlen(p)); p++; continue; }
                if (*p=='"' && (p[1]==',' || p[1]=='\0' || p[1]=='\r' || p[1]=='\n')) break;
                p++;
            }
            if (*p == '"') { *p = '\0'; p++; }
            if (*p == ',') { *p = '\0'; p++; }
        } else {
            out_fields[count++] = p;
            while (*p && *p != ',' && *p != '\r' && *p != '\n') p++;
            if (*p == ',') { *p = '\0'; p++; }
        }
        char *end = out_fields[count-1] + strlen(out_fields[count-1]);
        while (end > out_fields[count-1] && isspace((unsigned char)*(end-1))) end--;
        *end = '\0';
    }
    return count;
}

static ssize_t read_line_dynamic(FILE *f, char **buf, size_t *cap) {
    if (!*buf || *cap == 0) { *cap = 4096; *buf = (char*)malloc(*cap); if (!*buf) return -1; }
    size_t len = 0;
    for (;;) {
        if (!fgets(*buf + len, (int)(*cap - len), f)) { if (len == 0) return -1; break; }
        len += strlen(*buf + len);
        if (len > 0 && (*buf)[len-1] == '\n') break;
        size_t newcap = (*cap < 262144 ? (*cap * 2) : (*cap + 262144));
        char *nb = (char*)realloc(*buf, newcap); if (!nb) return -1;
        *buf = nb; *cap = newcap;
    }
    return (ssize_t)len;
}

static void write_output_header(FILE *fo) {
    static const char *feat_names[42] = {
        "voltage","measured","min_val","max_val","std_dev","avg","median",
        "bounce_back","drop","slope_bounce_back","slope_drop",
        "min_volt_below_19","max_volt_19_above","start_voltage","recovery_time_ms",
        "area_0_200ms","count_below7","count_below9","count_below10","curve_kurtosis",
        "curve_skew","max_rise_rate_0_180","max_fall_rate_0_180","mean_abs_slope_0_180",
        "std_slope_0_180","mean_abs_accel_0_180","max_accel_0_180","min_accel_0_180",
        "norm_energy_200ms","rec_slope","r_est","spike_cnt","dip_cnt",
        "prom_sum","spike_w_mean_ms","step_count_sust","max_step_mag",
        "bp_low","bp_mid","bp_high","bp_mid_ratio","bp_high_ratio"
    };
    fprintf(fo, "Test_Record_Detail_ID");
    for (int i=0;i<42;i++) fprintf(fo, ",%s", feat_names[i]);
    fprintf(fo, "\n");
}

static int is_blank_csv_line(const char *s) {
    if (!s) return 1;
    while (*s) { char c=*s++; if (c!=' '&&c!='\t'&&c!='\r'&&c!='\n'&&c!=',') return 0; }
    return 1;
}

static void write_row(FILE *fo, const char *id, const float features[42]) {
    fprintf(fo, "\"%s\"", id ? id : "");
    for (int i=0;i<42;i++) {
        if (isnan(features[i])) fprintf(fo, ",");
        else                    fprintf(fo, ",%.6f", features[i]);
    }
    fprintf(fo, "\n");
}

static void trim_id_inplace(char *s) {
    if (!s) return;
    char *p=s; while(*p && isspace((unsigned char)*p)) p++;
    if (p!=s) memmove(s,p,strlen(p)+1);
    size_t n=strlen(s);
    while (n && isspace((unsigned char)s[n-1])) s[--n]='\0';
    n=strlen(s);
    if (n>=2 && ((s[0]=='"' && s[n-1]=='"') || (s[0]=='\'' && s[n-1]=='\''))) {
        memmove(s, s+1, n-2); s[n-2]='\0';
    }
}

typedef struct { char **data; int count; int cap; } IdSet;
static void idset_init(IdSet *S){ S->data=NULL; S->count=0; S->cap=0; }
static int  idset_contains(IdSet *S,const char *id){ for(int i=0;i<S->count;i++) if(strcmp(S->data[i],id)==0) return 1; return 0; }
static int  idset_add(IdSet *S,const char *id){
    if (idset_contains(S,id)) return 0;
    if (S->count==S->cap){ int nc=S->cap?S->cap*2:64; char **nb=(char**)realloc(S->data,(size_t)nc*sizeof(char*));
        if(!nb) return -1; S->data=nb; S->cap=nc; }
    S->data[S->count]=strdup(id?id:""); if(!S->data[S->count]) return -1; S->count++; return 1;
}
static void idset_free(IdSet *S){ if(!S) return; for(int i=0;i<S->count;i++) free(S->data[i]); free(S->data); }

static int header_index(char **hdr, int n, const char **cands, int m) {
    for (int i=0;i<m;i++) for (int j=0;j<n;j++) if (iequals(hdr[j], cands[i])) return j;
    return -1;
}

// ---------------- main ----------------
int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr,"Usage: %s <input_csv_path>\n", argv[0]); return 1; }

    const char *input_path = argv[1];
    char *path_copy = strdup(input_path); if(!path_copy){ fprintf(stderr,"OOM\n"); return 1; }
    char *dir = dirname(path_copy);

    char out_path[4096];
    snprintf(out_path, sizeof(out_path), "%s/%s", dir, "MVT13_features_output.csv");
    free(path_copy);

    FILE *fi = fopen(input_path, "rb"); if(!fi){ fprintf(stderr,"Failed to open input: %s\n", input_path); return 1; }
    FILE *fo = fopen(out_path, "wb");  if(!fo){ fclose(fi); fprintf(stderr,"Failed to open output: %s\n", out_path); return 1; }

    char *line=NULL; size_t cap=0; ssize_t len;

    // header
    len = read_line_dynamic(fi, &line, &cap);
    if (len < 0) { fprintf(stderr, "Empty input CSV.\n"); fclose(fi); fclose(fo); free(line); return 1; }

    char *header_line = strdup(line); if(!header_line){ fclose(fi); fclose(fo); free(line); return 1; }
    char *fields[MAX_COLUMNS]; int nfields = split_csv_line(header_line, fields, MAX_COLUMNS);

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
        fclose(fi); fclose(fo); free(line); return 1;
    }

    write_output_header(fo);

    IdSet seen; idset_init(&seen);

    while ((len = read_line_dynamic(fi, &line, &cap)) >= 0) {
        if (len == 0 || is_blank_csv_line(line)) continue;

        char *row = strdup(line); if (!row) { fprintf(stderr,"OOM\n"); break; }
        char *cols[MAX_COLUMNS]; int nf = split_csv_line(row, cols, MAX_COLUMNS);
        if (nf <= 0 || (nf == 1 && (!cols[0] || cols[0][0] == '\0'))) { free(row); continue; }

        const char *id_raw   = (col.idx_id       < nf) ? cols[col.idx_id]       : "";
        const char *starter  = (col.idx_starter  < nf) ? cols[col.idx_starter]  : "";
        const char *swver    = (col.idx_swver    < nf) ? cols[col.idx_swver]    : "";
        const char *vstr     = (col.idx_voltage  < nf) ? cols[col.idx_voltage]  : "";
        const char *mstr     = (col.idx_measured < nf) ? cols[col.idx_measured] : "";
        const char *ripple   = (col.idx_ripple   < nf) ? cols[col.idx_ripple]   : "";

        char idbuf[1024]; snprintf(idbuf,sizeof(idbuf), "%s", id_raw?id_raw:""); trim_id_inplace(idbuf);
        if (idbuf[0] == '\0') { free(row); continue; }

        float feats[42]; for (int i=0;i<42;i++) feats[i]=NAN;

        char *endp=NULL;
        float voltage = strtof(vstr, &endp); if (endp == vstr) { free(row); continue; }
        long mtmp = strtol(mstr, &endp, 10); if (endp == mstr) { free(row); continue; }
        int16_t measured = (int16_t)mtmp;

        float ripple_vals[MAX_ALTERNATOR_RIPPLE];
        int ripple_count = process_alternator_ripple(ripple, ripple_vals);
        float ripple_sum = sum_alternator_ripple_array(ripple_vals, ripple_count);
        (void)ripple_sum; // placeholder—keep if you’ll feed the model later

        Record rec;
        preprocess_record(starter, swver, voltage, (float)measured, &rec);
        pack_features(&rec, feats);

        if (idset_add(&seen, idbuf) < 0) { free(row); break; }
        write_row(fo, idbuf, feats);

        free(row);
    }

    idset_free(&seen);
    free(line); fclose(fi); fclose(fo);
    return 0;
}
