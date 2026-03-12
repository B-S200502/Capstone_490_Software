#ifndef USB_DEBUG_H
#define USB_DEBUG_H

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
typedef struct {
    uint32_t count;
    uint32_t avg_window;
    uint32_t sum_p2p_raw[4][4];

    uint32_t print_period_ms;
    uint32_t last_print_ms;
} UsbDbgP2P_t;

/* =========================================================================
 * FUNCTION PROTOTYPES
 * ========================================================================= */

int usb_printf(const char *fmt, ...);

/**
 * @brief Initialize USB debug statistics/printing
 * @param avg_window         number of prints to average over (e.g., 16)
 * @param print_period_ms    throttle period (e.g., 200ms)
 */
void usb_dbg_init(uint32_t avg_window, uint32_t print_period_ms);

/**
 * @brief Push one synchronized window (all 4 ADCs) and print clean P2P stats (rate-limited)
 * @param adc1..adc4         pointers to interleaved 4-ch samples (CH0..CH3) for each ADC
 * @param num_samples        number of sample-frames (e.g., 1024). Each frame has 4 uint16_t.
 * @param adc_lsb_volts      volts per LSB (e.g., 3.3f/4095.0f)
 */
void usb_dbg_push_adc_window_4adc(const uint16_t *adc1,
                                  const uint16_t *adc2,
                                  const uint16_t *adc3,
                                  const uint16_t *adc4,
                                  uint32_t num_samples);

void usb_dbg_stream_mic_window_csv(uint8_t mic_index,
                                   const uint16_t *adc_buf,
                                   uint8_t adc_channel,
                                   uint32_t num_samples,
                                   uint32_t sample_rate_hz);

#ifdef __cplusplus
}
#endif

#endif /* USB_DEBUG_H */