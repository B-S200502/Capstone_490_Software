/**
 * @file dsp_pipeline_test.c
 * @brief Unit tests for DSP pipeline functions
 * @details This file contains unit tests for the DSP pipeline functions defined in dsp_pipeline.c,
 */

/* =========================================================================
 * INCLUDES
 * ========================================================================= */
#include <stdint.h>
#include <math.h>

#include "arm_math.h"
#include "usb/usb_debug.h"
#include "app_main.h"       
#include "dsp_pipeline.h"
#include "dsp_pipeline_test.h"
/* =========================================================================
 * DEFINES & CONSTANTS
 * ========================================================================= */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TEST_FS_HZ        (48000.0f)
#define TEST_TONE_HZ      (3000.0f)

#define TEST_TONE1_HZ  937.5f   // bin 20 at fs=48k, N=1024
#define TEST_TONE2_HZ  2343.75f // bin 50
#define TEST_TONE3_HZ  4687.5f  // bin 100

// Print settings
#define PRINT_MAX_BINS    56   // print bins 0..55
#define PRINT_PEAK_SCAN   512 // scan first N/2 bins for peak


/* =========================================================================
 * STATIC VARIABLES
 * ========================================================================= */

static float s_time[FRAME_SIZE];
static float s_rfft_packed[FRAME_SIZE];         // CMSIS packed output length = N floats
static float s_mag[(FRAME_SIZE / 2) + 1];       // one-sided magnitude length = N/2 + 1

/* =========================================================================
 * FORWARD DECLARATIONS
 * ========================================================================= */

static void dsp_gen_sine(float *buf, float freq, float fs, uint32_t N);
static void dsp_gen_multi_sine(float *buf,
                               float f1, float a1,
                               float f2, float a2,
                               float f3, float a3,
                               float fs,
                               uint32_t N);
// static void dsp_gen_impulse(float *buf, uint32_t N);
// static void dsp_gen_noise(float *buf, uint32_t N);
// static void dsp_gen_dc(float *buf, float value, uint32_t N);

static void dsp_validate_single_sine(float freq, 
                              float fs, 
                              uint32_t N, 
                              uint32_t k_peak);
static void dsp_validate_multi_sine(float fs,
                                    uint32_t N,
                                    float *mag,
                                    float f1,
                                    float f2,
                                    float f3);

static void dsp_test_single_sine(void);
static void dsp_test_multi_sine(void);
/* static void module_init(void); */
/* static void module_process(void); */

/* =========================================================================
 * PUBLIC FUNCTIONS
 * ========================================================================= */

void rfft_packed_to_mag(const float *packed, float *mag, uint32_t N)
{
  const uint32_t half = N >> 1; // N/2

  mag[0]    = fabsf(packed[0]); // DC = Re(X[0])
  mag[half] = fabsf(packed[1]); // Nyquist = Re(X[N/2]) for even N

  for (uint32_t k = 1; k < half; k++) {
      float re = packed[2U * k];
      float im = packed[2U * k + 1U];
      mag[k] = sqrtf(re * re + im * im);
  }
}

void dsp_unit_test_suite(void) {
  // Single sine test
  dsp_test_single_sine();
  dsp_test_multi_sine();

  //dsp_run_fft_unit_test();
}

void dsp_analyze_signal(float *time, float *fft_buf, float *mag, uint32_t N) {

  // Init FFT
  arm_rfft_fast_instance_f32 fft;
  arm_status st = arm_rfft_fast_init_f32(&fft, N);
  if (st != ARM_MATH_SUCCESS) {
    usb_printf("arm_rfft_fast_init_f32 FAILED (status=%d)\r\n", (int)st);
    return;
  }

  // Run RFFT (forward)
  arm_rfft_fast_f32(&fft, s_time, s_rfft_packed, 0);

  // Magnitude (one-sided)
  rfft_packed_to_mag(s_rfft_packed, s_mag, N);
}

uint32_t dsp_print_fft_report(float fs, float *mag, uint32_t N) {
  const uint32_t half = N >> 1;
  const float bin_res = fs / (float)N;

  usb_printf("fs=%d.%d Hz, N=%lu, bin_res=%d.%03d Hz/bin\r\n",
             PRINT_F1(fs),
             (unsigned long)N,
             PRINT_F3(bin_res));

  // Find peak bin
  uint32_t k_peak = 0;
  float peak = 0.0f;

  for (uint32_t k = 0; k <= half; k++) {
    if (mag[k] > peak) {
      peak = mag[k];
      k_peak = k;
    }
  }

  usb_printf("peak_bin=%lu, peak_mag=%d.%03d\r\n",
            (unsigned long)k_peak,
            PRINT_F3(peak));
  
  // Print first bins
  usb_printf("\r\nBin\tFreq(Hz)\tMag\r\n");

  uint32_t max_bins = PRINT_MAX_BINS;
  if (max_bins > (half + 1U))
    max_bins = half + 1U;

  for (uint32_t k = 0; k < max_bins; k++) {
    float f = (float)k * bin_res;

    usb_printf("%lu\t%d.%d\t\t%d.%03d\r\n",
               (unsigned long)k,
               PRINT_F1(f),
               PRINT_F3(mag[k]));
  }

  // Peak neighborhood
  usb_printf("\r\nPeak neighborhood:\r\n");

  uint32_t k0 = (k_peak > 5U) ? (k_peak - 5U) : 0U;
  uint32_t k1 = (k_peak + 5U <= half) ? (k_peak + 5U) : half;

  for (uint32_t k = k0; k <= k1; k++) {
    float f = (float)k * bin_res;

    usb_printf("%lu\t%d.%d\t\t%d.%03d%s\r\n",
               (unsigned long)k,
               PRINT_F1(f),
               PRINT_F3(mag[k]),
               (k == k_peak) ? "  <-- PEAK" : "");
  }

  return k_peak;
}

void dsp_run_fft_unit_test(void) {

  // Init FFT
  arm_rfft_fast_instance_f32 fft;
  arm_status st = arm_rfft_fast_init_f32(&fft, FRAME_SIZE);
  if (st != ARM_MATH_SUCCESS) {
    usb_printf("arm_rfft_fast_init_f32 FAILED (status=%d)\r\n", (int)st);
    return;
  }

  // Run RFFT (forward)
  arm_rfft_fast_f32(&fft, s_time, s_rfft_packed, 0);

  // Magnitude (one-sided)
  rfft_packed_to_mag(s_rfft_packed, s_mag, FRAME_SIZE);

  // Find peak bin
  const uint32_t half = FRAME_SIZE >> 1;
  uint32_t k_peak = 0;
  float peak = 0.0f;

  for (uint32_t k = 0; k <= half; k++) {
    if (s_mag[k] > peak) {
      peak = s_mag[k];
      k_peak = k;
    }
  }

  // Expected bin
  const float bin_res = TEST_FS_HZ / (float)FRAME_SIZE;
  const float k_expected = TEST_TONE_HZ / bin_res;

  usb_printf("fs=%d.%d Hz, N=%lu, bin_res=%d.%03d Hz/bin\r\n",
             PRINT_F1(TEST_FS_HZ),
             (unsigned long)FRAME_SIZE,
             PRINT_F3(bin_res));

  usb_printf("tone=%d.%d Hz, expected_bin=%d.%02d\r\n",
             PRINT_F1(TEST_TONE_HZ),
             PRINT_F2(k_expected));

  usb_printf("peak_bin=%lu, peak_mag=%d.%03d\r\n",
             (unsigned long)k_peak,
             PRINT_F3(peak));

  // Print first bins
  usb_printf("\r\nBin\tFreq(Hz)\tMag\r\n");

  uint32_t max_bins = PRINT_MAX_BINS;
  if (max_bins > (half + 1U))
    max_bins = half + 1U;

  for (uint32_t k = 0; k < max_bins; k++) {

    float f = (float)k * bin_res;

    usb_printf("%lu\t%d.%d\t\t%d.%03d\r\n",
               (unsigned long)k,
               PRINT_F1(f),
               PRINT_F3(s_mag[k]));
  }

  // Peak neighborhood
  usb_printf("\r\nPeak neighborhood:\r\n");

  uint32_t k0 = (k_peak > 5U) ? (k_peak - 5U) : 0U;
  uint32_t k1 = (k_peak + 5U <= half) ? (k_peak + 5U) : half;

  for (uint32_t k = k0; k <= k1; k++) {
    float f = (float)k * bin_res;

    usb_printf("%lu\t%d.%d\t\t%d.%03d%s\r\n",
               (unsigned long)k,
               PRINT_F1(f),
               PRINT_F3(s_mag[k]),
               (k == k_peak) ? "  <-- PEAK" : "");
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

/* --- SIGNAL GENERATORS --- */

static void dsp_gen_sine(float *buf, float freq, float fs, uint32_t N) {

  // Generate sine: x[n] = sin(2*pi*f*n/fs)
  const float w = 2.0f * (float)M_PI * (freq / fs);

  for (uint32_t n = 0; n < FRAME_SIZE; n++) {
    buf[n] = arm_sin_f32(w * (float)n);
  }
}

static void dsp_gen_multi_sine(float *buf,
                               float f1, float a1,
                               float f2, float a2,
                               float f3, float a3,
                               float fs,
                               uint32_t N) {
  const float w1 = 2.0f * (float)M_PI * (f1 / fs);
  const float w2 = 2.0f * (float)M_PI * (f2 / fs);
  const float w3 = 2.0f * (float)M_PI * (f3 / fs);

  for (uint32_t n = 0; n < N; n++) {
    buf[n] = a1 * arm_sin_f32(w1 * (float)n)
           + a2 * arm_sin_f32(w2 * (float)n)
           + a3 * arm_sin_f32(w3 * (float)n);
  }
}

/* --- VALIDATION FUNCTIONS --- */

static void dsp_validate_single_sine(float freq, float fs, uint32_t N, uint32_t k_peak) {

  // Expected bin
  const float bin_res = fs / (float)N;
  const float k_expected = freq / bin_res;

  usb_printf("tone=%d.%d Hz, expected_bin=%d.%02d\r\n",
             PRINT_F1(freq),
             PRINT_F2(k_expected));

  uint32_t k_expected_round = (uint32_t)(k_expected + 0.5f);
  if (k_expected_round == k_peak) {
    usb_printf("Single sine test: PASS\r\n");
  } else {
    usb_printf("Single sine test: FAIL\r\n");
  }
}

static void dsp_validate_multi_sine(float fs,
                                    uint32_t N,
                                    float *mag,
                                    float f1,
                                    float f2,
                                    float f3) {
  const float bin_res = fs / (float)N;

  uint32_t k1 = (uint32_t)((f1 / bin_res) + 0.5f);
  uint32_t k2 = (uint32_t)((f2 / bin_res) + 0.5f);
  uint32_t k3 = (uint32_t)((f3 / bin_res) + 0.5f);

  usb_printf("expected bins: %lu, %lu, %lu\r\n",
             (unsigned long)k1,
             (unsigned long)k2,
             (unsigned long)k3);

  usb_printf("magnitudes: %d.%03d, %d.%03d, %d.%03d\r\n",
             PRINT_F3(mag[k1]),
             PRINT_F3(mag[k2]),
             PRINT_F3(mag[k3]));
}

/* --- UNIT TESTS --- */

static void dsp_test_single_sine() {
  usb_printf("\r\n--- DSP UNIT TEST: SINE -> RFFT -> MAG ---\r\n");
  // Generate test sine wave
  dsp_gen_sine(s_time, TEST_TONE_HZ, TEST_FS_HZ, FRAME_SIZE);

  // Analyze signal (RFFT + MAG)
  dsp_analyze_signal(s_time, s_rfft_packed, s_mag, FRAME_SIZE);

  uint32_t k_peak = dsp_print_fft_report(TEST_FS_HZ, s_mag, FRAME_SIZE);

  dsp_validate_single_sine(TEST_TONE_HZ, TEST_FS_HZ, FRAME_SIZE, k_peak);

  usb_printf("--- END DSP UNIT TEST ---\r\n");
}

static void dsp_test_multi_sine(void) {
  usb_printf("\r\n--- DSP UNIT TEST: MULTI-SINE -> RFFT -> MAG ---\r\n");

  dsp_gen_multi_sine(s_time,
                     TEST_TONE1_HZ, 1.0f,
                     TEST_TONE2_HZ, 0.7f,
                     TEST_TONE3_HZ, 0.4f,
                     TEST_FS_HZ,
                     FRAME_SIZE);

  dsp_analyze_signal(s_time, s_rfft_packed, s_mag, FRAME_SIZE);

  dsp_print_fft_report(TEST_FS_HZ, s_mag, FRAME_SIZE);

  dsp_validate_multi_sine(TEST_FS_HZ,
                          FRAME_SIZE,
                          s_mag,
                          TEST_TONE1_HZ,
                          TEST_TONE2_HZ,
                          TEST_TONE3_HZ);

  usb_printf("--- END DSP UNIT TEST ---\r\n");
}
/**
 * @brief Static helper function description
 * @return void
 */
