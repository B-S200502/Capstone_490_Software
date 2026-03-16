/**
 * @file template.c
 * @brief Brief description of module functionality
 * @details Detailed description of what this module does and its responsibilities
 */

/* =========================================================================
 * INCLUDES
 * ========================================================================= */

#include <stdint.h>

#include "timing.h"

/* DWT (Data Watchpoint and Trace) cycle counter support */
#include "stm32g4xx_hal.h"

/* =========================================================================
 * DEFINES & CONSTANTS
 * ========================================================================= */

 
/* FFT Performance Measurement */
#define FFT_PERF_BETA 0.1f

/* =========================================================================
 * STATIC VARIABLES
 * ========================================================================= */

 /* FFT Performance measurement variables */
static float fft_precise_average = 0.0f;
static float fft_last_cycles = 0.0f;
static uint8_t fft_perf_initialized = 0;

static uint32_t start_cycles = 0;
static uint32_t end_cycles = 0;

/* =========================================================================
 * FORWARD DECLARATIONS
 * ========================================================================= */

/* static void module_init(void); */
/* static void module_process(void); */

/* =========================================================================
 * PUBLIC FUNCTIONS
 * ========================================================================= */

void init_fft_performance_measurement(void) {
  // Enable DWT (Data Watchpoint and Trace)
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  
  // Reset and enable DWT cycle counter
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  
  // Initialize performance tracking
  fft_precise_average = 0.0f;
  fft_last_cycles = 0.0f;
  fft_perf_initialized = 1;
}

float get_fft_avg_cycles(void)
{
  return fft_precise_average;
}

float get_fft_last_cycles(void)
{
  return fft_last_cycles;
}

/**
 * @brief Keeps a moving average of clock cycles taken by code block
 * @param[in] cycles cycles taken by most recent code block execution
 * @return Description of return value
 */
void calculate_fft_cycles_average(float cycles)
{
    float lpVal = fft_precise_average;
    
    fft_precise_average = (lpVal - (FFT_PERF_BETA * ((float)(lpVal - cycles))));
    fft_last_cycles = cycles;
}

void start_performance_measurement(void)
{
  // Measure FFT performance using DWT cycle counter
  start_cycles = DWT->CYCCNT;
}

void end_performance_measurement(void)
{
  end_cycles = DWT->CYCCNT;
  uint32_t elapsed_cycles = end_cycles - start_cycles;
  calculate_fft_cycles_average((float)elapsed_cycles);
}

/* ============================================================================
 * STATIC FUNCTIONS
 * ============================================================================ */

/**
 * @brief Static helper function description
 * @return void
 */
