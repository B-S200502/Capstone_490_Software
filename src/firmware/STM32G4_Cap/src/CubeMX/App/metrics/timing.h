#ifndef TEMPLATE_H
#define TEMPLATE_H

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * INCLUDES
 * ========================================================================= */

/* =========================================================================
 * DEFINES
 * ========================================================================= */


/* =========================================================================
 * TYPE DEFINITIONS
 * ========================================================================= */

/* =========================================================================
 * FUNCTION PROTOTYPES
 * ========================================================================= */

void init_fft_performance_measurement(void);
float get_fft_avg_cycles(void);
float get_fft_last_cycles(void);
void calculate_fft_cycles_average(float cycles);
void start_performance_measurement(void);
void end_performance_measurement(void);

#ifdef __cplusplus
}
#endif

#endif /* TEMPLATE_H */