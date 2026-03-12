/**
 * @file dsp_pipeline.c
 * @brief Contains DSP functions for ADC audio signals including FFT processing
 * @details This module is responsible for processing raw ADC data into the frequency domain using FFT, as well as any necessary pre-processing steps like DC offset removal and normalization.
 */

/* =========================================================================
 * INCLUDES
 * ========================================================================= */
#include <stdint.h>
#include <math.h>

#include "arm_math.h"

#include "app_main.h"
#include "dsp_pipeline.h"

/* =========================================================================
 * DEFINES & CONSTANTS
 * ========================================================================= */

/* =========================================================================
 * STATIC VARIABLES
 * ========================================================================= */
static const float adc_scalar = 3.3f / 4095.0f;

static float fft_input_buf[FRAME_SIZE];
static float fft_temp_buf[2048];

/* =========================================================================
 * FORWARD DECLARATIONS
 * ========================================================================= */

/* static void module_init(void); */
/* static void module_process(void); */
static void pack_rfft_complex_bins(const float *packed_fft,
                                   float *complex_output,
                                   uint32_t fft_size);

/* =========================================================================
 * PUBLIC FUNCTIONS
 * ========================================================================= */

// Process raw ADC data to float voltage values for a specific channel 
// (de-interleaves)
void process_adc_to_float(uint16_t *adc_raw, float *output, uint8_t ch, 
                          uint32_t length) {
  for (uint32_t i = 0; i < length; i++) {
      output[i] = (float)adc_raw[i * N_CH_PER_ADC + ch] * adc_scalar;
  }
}

float calculate_dc_offset(float *data, uint32_t length)
{
  float sum = 0.0f;
  for (uint32_t i = 0; i < length; i++) {
      sum += data[i];
  }
  return sum / (float)length;
}

void remove_dc_bias(float *data, uint32_t length, float dc_offset)
{
  for (uint32_t i = 0; i < length; i++) {
      data[i] -= dc_offset;
  }
}

void apply_fft(arm_rfft_fast_instance_f32 *fft_instance, 
               float *input, 
               float *complex_output, 
               uint32_t fft_size) {
  arm_rfft_fast_f32(fft_instance, input, fft_temp_buf, 0);
  pack_rfft_complex_bins(fft_temp_buf, complex_output, fft_size);
}

// TODO: We likely don't need to calculate magnitude, I think the 
// beamforming algorithm can work directly with complex FFT output.
void calculate_magnitude(float *fft_output, float *magnitude, uint32_t fft_size)
{
  // For real FFT output of size N, the bins are arranged as:
  // [Re(0), Re(N/2), Re(1), Im(1), Re(2), Im(2), ..., Re(N/2-1), Im(N/2-1)]

  const uint32_t half = fft_size >> 1; // N/2

  // Bin 0 and N/2 are purely real.
  magnitude[0]    = fabsf(fft_output[0]);  // DC bin
  magnitude[half] = fabsf(fft_output[1]);  // Nyquist bin

  for (uint32_t i = 1; i < half; i++) {
    float re = fft_output[2 * i];
    float im = fft_output[2 * i + 1];
    magnitude[i] = sqrtf(re * re + im * im);
  } 
}

void normalize_magnitude(float *mag, uint32_t length)
{
  float max = 0.0f;
  for (uint32_t i = 0; i < length; i++) {
      if (mag[i] > max) max = mag[i];
  }
  
  if (max > 0.0f) {
      for (uint32_t i = 0; i < length; i++) {
          mag[i] /= max;
      }
  }
}

void process_adc_channel_pipeline(arm_rfft_fast_instance_f32 *fft_instance,
                                  uint16_t *adc_raw,
                                  uint8_t channel_index,
                                  float *fft_output)
{
  process_adc_to_float(adc_raw, fft_input_buf, channel_index, FRAME_SIZE);
  float dc_offset = calculate_dc_offset(fft_input_buf, FRAME_SIZE);
  remove_dc_bias(fft_input_buf, FRAME_SIZE, dc_offset);
  apply_fft(fft_instance, fft_input_buf, fft_output, FRAME_SIZE);
}

void process_adc_pipeline(arm_rfft_fast_instance_f32 *fft_instance,
                          uint16_t *adc_raw, 
                          uint32_t adc_id, 
                          float *fft_output_base)
{
  // When a half-buffer is ready, 4 channels of interleaved data are available.
  // Rather than de-interleaving all 4 channels into separate buffers, we can 
  // process each channel is a scratch buffer.
  (void)adc_id;
  for (uint8_t ch = 0; ch < N_CH_PER_ADC; ch++) {
  
    float *fft_output = fft_output_base + ch * (2 * N_BINS);
    process_adc_channel_pipeline(fft_instance, adc_raw, ch, fft_output);

    // normalize_magnitude(fft_output, 512);
  }
}

/**
 * @brief Public function description
 * @param[in] param1 Description of input parameter
 * @return Description of return value
 */

/* ============================================================================
 * STATIC FUNCTIONS
 * ============================================================================ */

/**
 * @brief Static helper function description
 * @return void
 */
static void pack_rfft_complex_bins(const float *packed_fft,
                                   float *complex_output,
                                   uint32_t fft_size)
{
  const uint32_t half = fft_size >> 1;

  complex_output[0] = packed_fft[0];
  complex_output[1] = 0.0f;

  for (uint32_t bin = 1; bin < half; bin++) {
    complex_output[2u * bin] = packed_fft[2u * bin];
    complex_output[2u * bin + 1u] = packed_fft[2u * bin + 1u];
  }

  complex_output[2u * half] = packed_fft[1];
  complex_output[2u * half + 1u] = 0.0f;
}
