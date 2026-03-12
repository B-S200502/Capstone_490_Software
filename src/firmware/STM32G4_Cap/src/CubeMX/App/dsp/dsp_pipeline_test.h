#ifndef DSP_PIPELINE_TEST_H
#define DSP_PIPELINE_TEST_H

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * INCLUDES
 * ========================================================================= */

#include <stdint.h>
/* =========================================================================
 * DEFINES
 * ========================================================================= */

/* =========================================================================
 * TYPE DEFINITIONS
 * ========================================================================= */

/* =========================================================================
 * FUNCTION PROTOTYPES
 * ========================================================================= */

void rfft_packed_to_mag(const float *packed, float *mag, uint32_t N);
void dsp_unit_test_suite(void);
void dsp_analyze_signal(float *time, float *fft_buf, float *mag, uint32_t N);
uint32_t dsp_print_fft_report(float fs, float *mag, uint32_t N);

void dsp_run_fft_unit_test(void);







#ifdef __cplusplus
}
#endif

#endif /* DSP_PIPELINE_TEST_H */