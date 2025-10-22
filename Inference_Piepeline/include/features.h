#ifndef FEATURES_H_
#define FEATURES_H_

#include <stdint.h>
#include "record.h"
#include "numpy_functions.h"

/** Compute all engineered features for a single trace. */
void compute_features(const float fvalues[], int16_t n,
                      Record *out, float voltage, int16_t measured);

#endif // FEATURES_H_
