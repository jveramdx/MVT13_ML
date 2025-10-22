// test_numpy_functions.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <ctype.h>

#include "numpy_functions.h"
#include "features.h"

#define MAX_SAMPLES 50
#define LINE_BUFFER 65536
#define DATASET_PATH "../Preprocessed_Datasets/MVT13_Starter_Voltage_Array.csv"

typedef struct {
    long long record_id;
    int length;
    float values[FEATURES_MAX_POINTS];
} Sample;

static bool parse_float_token(const char *token, float *out) {
    if (!token) return false;
    while (*token && isspace((unsigned char)*token)) token++;
    if (!*token) return false;
    char *endptr = NULL;
    float val = strtof(token, &endptr);
    if (endptr == token) return false;
    while (*endptr && isspace((unsigned char)*endptr)) endptr++;
    if (*endptr != '\0') return false;
    *out = val;
    return true;
}

static int load_samples(const char *path, Sample *samples, int max_samples) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open dataset: %s\n", path);
        return -1;
    }

    char line[LINE_BUFFER];
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return -1;
    }

    int count = 0;
    while (count < max_samples && fgets(line, sizeof(line), fp)) {
        Sample *s = &samples[count];
        s->record_id = 0;
        s->length = 0;

        char *token = strtok(line, ",\r\n");
        if (!token) continue;
        s->record_id = atoll(token);

        int idx = 0;
        while ((token = strtok(NULL, ",\r\n")) != NULL) {
            float value;
            if (!parse_float_token(token, &value)) continue;
            if (idx < FEATURES_MAX_POINTS) {
                s->values[idx++] = value;
            }
        }
        s->length = idx;
        count++;
    }

    fclose(fp);
    return count;
}

static void print_float_sequence(const float *data, int n) {
    for (int i = 0; i < n; ++i) {
        if (i > 0) putchar(';');
        printf("%.9g", (double)data[i]);
    }
}

static void print_complex_sequence(const float _Complex *data, int n) {
    for (int i = 0; i < n; ++i) {
        if (i > 0) putchar(';');
        float re = crealf(data[i]);
        float im = cimagf(data[i]);
        printf("%.9g%+.9gi", (double)re, (double)im);
    }
}

static void print_int_sequence(const int32_t *data, int n) {
    for (int i = 0; i < n; ++i) {
        if (i > 0) putchar(';');
        printf("%d", (int)data[i]);
    }
}

static void run_fft(const Sample *samples, int count) {
    for (int i = 0; i < count; ++i) {
        const Sample *s = &samples[i];
        if (s->length <= 0) continue;
        int n = s->length;
        float _Complex *input = (float _Complex *)malloc(sizeof(float _Complex) * (size_t)n);
        if (!input) continue;
        for (int j = 0; j < n; ++j) {
            input[j] = s->values[j];
        }
        float _Complex *out = np_fft_fft(input, n, n, FFT_NORM_BACKWARD, NULL);
        free(input);
        if (!out) continue;
        printf("fft,%d,%lld,", i, s->record_id);
        print_complex_sequence(out, n);
        printf("\n");
        free(out);
    }
}

static void run_rfftfreq(const Sample *samples, int count) {
    for (int i = 0; i < count; ++i) {
        const Sample *s = &samples[i];
        if (s->length <= 0) continue;
        int n = s->length;
        float *freqs = np_fft_rfftfreq(n, 0.01f); // 10 ms sampling step
        if (!freqs) continue;
        int out_len = (n % 2 == 0) ? (n / 2 + 1) : ((n + 1) / 2);
        printf("rfftfreq,%d,%lld,", i, s->record_id);
        print_float_sequence(freqs, out_len);
        printf("\n");
        free(freqs);
    }
}

static void run_find_peaks(const Sample *samples, int count) {
    for (int i = 0; i < count; ++i) {
        const Sample *s = &samples[i];
        if (s->length <= 0) continue;
        peak_result_t peaks = find_peaks_ref(s->values, s->length,
                                             NULL, NULL,
                                             1, NULL, NULL,
                                             0, 0.5f, NULL);
        printf("find_peaks,%d,%lld,", i, s->record_id);
        if (peaks.n_peaks > 0 && peaks.indices) {
            print_int_sequence(peaks.indices, peaks.n_peaks);
        }
        printf("\n");
        free_peak_result(&peaks);
    }
}

static void run_medfilt(const Sample *samples, int count) {
    for (int i = 0; i < count; ++i) {
        const Sample *s = &samples[i];
        if (s->length <= 0) continue;
        int n = s->length;
        int k = (n >= 5) ? 5 : (n % 2 ? n : (n > 0 ? n - 1 : 1));
        if (k < 1) k = 1;
        if ((k & 1) == 0) k += 1;
        if (k > n) k = (n > 0) ? ((n % 2) ? n : n - 1) : 1;
        if (k < 1) k = 1;
        int32_t dims[1] = { n };
        int32_t kernel[1] = { k };
        float *out = (float *)malloc(sizeof(float) * (size_t)n);
        if (!out) continue;
        medfilt_nd(s->values, dims, 1, kernel, out);
        printf("medfilt,%d,%lld,", i, s->record_id);
        print_float_sequence(out, n);
        printf("\n");
        free(out);
    }
}

int main(int argc, char **argv) {
    const char *which = (argc > 1) ? argv[1] : "all";
    bool do_fft = false, do_rfftfreq = false, do_find_peaks = false, do_medfilt = false;

    if (strcmp(which, "all") == 0) {
        do_fft = do_rfftfreq = do_find_peaks = do_medfilt = true;
    } else if (strcmp(which, "fft") == 0) {
        do_fft = true;
    } else if (strcmp(which, "rfftfreq") == 0) {
        do_rfftfreq = true;
    } else if (strcmp(which, "find_peaks") == 0) {
        do_find_peaks = true;
    } else if (strcmp(which, "medfilt") == 0) {
        do_medfilt = true;
    } else {
        fprintf(stderr, "Unknown method '%s'. Valid options: all, fft, rfftfreq, find_peaks, medfilt.\n", which);
        return 1;
    }

    Sample samples[MAX_SAMPLES];
    int sample_count = load_samples(DATASET_PATH, samples, MAX_SAMPLES);
    if (sample_count <= 0) {
        fprintf(stderr, "No samples found in dataset.\n");
        return 1;
    }

    if (do_fft)       run_fft(samples, sample_count);
    if (do_rfftfreq)  run_rfftfreq(samples, sample_count);
    if (do_find_peaks)run_find_peaks(samples, sample_count);
    if (do_medfilt)   run_medfilt(samples, sample_count);

    return 0;
}
