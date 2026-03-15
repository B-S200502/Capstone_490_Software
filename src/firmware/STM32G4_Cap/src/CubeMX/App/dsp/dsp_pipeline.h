#ifndef DSP_PIPELINE_H
#define DSP_PIPELINE_H

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * INCLUDES
 * ========================================================================= */

#include <stdint.h>

#include "arm_math.h"

/* DWT (Data Watchpoint and Trace) cycle counter support */
#include "stm32g4xx_hal.h"

/* =========================================================================
 * DEFINES
 * ======================================================================== */

/* FFT Performance Measurement */
#define FFT_PERF_BETA 0.1f

/* FFT Bin Averaging (EMA) */
#define FFT_BIN_AVG_BETA 0.1f

/* =========================================================================
 * TYPE DEFINITIONS
 * ======================================================================== */

/* =========================================================================
 * FUNCTION PROTOTYPES
 * ========================================================================= */

void process_adc_channel_pipeline(arm_rfft_fast_instance_f32 *fft_instance,
                                  uint16_t *adc_raw,
                                  uint8_t channel_index,
                                  float *fft_output);

void process_adc_pipeline(arm_rfft_fast_instance_f32 *fft_instance,
                          uint16_t *adc_raw, 
                          uint32_t adc_id, 
                          float *fft_output);

/* Performance measurement functions */
void init_fft_performance_measurement(void);
float get_fft_avg_cycles(void);
float get_fft_last_cycles(void);

/* FFT bin averaging (element-wise EMA across windows) */
void update_fft_bin_average(float *avg, const float *new_data, uint32_t length, float beta);

#ifdef __cplusplus
}
#endif

#endif /* DSP_PIPELINE_H */