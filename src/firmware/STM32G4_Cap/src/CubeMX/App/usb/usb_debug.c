/**
 * @file usb_debug.c
 * @brief USB CDC debug utilities (printf + ADC P2P reporting)
 * @details Provides robust CDC printing and a P2P (peak-to-peak) report for 4 ADCs x 4 channels.
 */

/* =========================================================================
 * INCLUDES
 * ========================================================================= */
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "usb_debug.h"
#include "usbd_cdc_if.h"
#include "stm32g4xx_hal.h"
#include "app_main.h"



/* =========================================================================
 * DEFINES & CONSTANTS
 * ========================================================================= */
#define USB_DBG_NUM_ADC        4u
#define USB_DBG_NUM_CH         4u
#define USB_DBG_CDC_CHUNK      64u

// If your ADC reference/resolution changes, update here.
// (Using integers avoids float printf dependencies.)
#define USB_DBG_ADC_VREF_MV    2900u
#define USB_DBG_ADC_MAX_COUNTS 4095u

/* =========================================================================
 * STATIC VARIABLES
 * ========================================================================= */
typedef struct {
    uint32_t count;
    uint32_t avg_window;
    uint32_t sum_p2p_raw[USB_DBG_NUM_ADC][USB_DBG_NUM_CH];

    uint32_t print_period_ms;
    uint32_t last_print_ms;
} UsbDbgState_t;

static UsbDbgState_t g_dbg;

/* =========================================================================
 * FORWARD DECLARATIONS
 * ========================================================================= */
static void usb_cdc_write_chunked_len(const uint8_t *data, uint32_t len);
static void compute_p2p_raw_adc4ch(const uint16_t *adc_buf,
                                  uint32_t num_samples,
                                  uint16_t p2p_out_raw[USB_DBG_NUM_CH]);
static void usb_dbg_flush_string(char *buffer, size_t *length);

/* =========================================================================
 * PUBLIC FUNCTIONS
 * ========================================================================= */

int usb_printf(const char *fmt, ...)
{
    char buffer[512];
    va_list args;
    va_start(args, fmt);
    int len = vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    if (len < 0) {
        // Encoding error
        return len;
    } else if (len >= (int)sizeof(buffer)) {
        // Output was truncated
        len = sizeof(buffer) - 1; // Adjust to actual buffer size
    }

#if MODE == RELEASE
    // Best-effort send; drop if USB busy
    if (CDC_Transmit_FS((uint8_t*)buffer, (uint16_t)len) == USBD_OK) {
        return len;
    }
#else
    // Wait until USB ready
    while (CDC_Transmit_FS((uint8_t*)buffer, (uint16_t)len) == USBD_BUSY)
    {
        HAL_Delay(1);  // small wait
    }
#endif

    return len;
}

void usb_dbg_init(uint32_t avg_window, uint32_t print_period_ms)
{
    memset(&g_dbg, 0, sizeof(g_dbg));
    g_dbg.avg_window = (avg_window == 0u) ? 1u : avg_window;
    g_dbg.print_period_ms = (print_period_ms == 0u) ? 200u : print_period_ms;
    g_dbg.last_print_ms = 0u;
}

void usb_dbg_push_adc_window_4adc(const uint16_t *adc1,
                                  const uint16_t *adc2,
                                  const uint16_t *adc3,
                                  const uint16_t *adc4,
                                  uint32_t num_samples)
{
    uint32_t now = HAL_GetTick();
    if ((now - g_dbg.last_print_ms) < g_dbg.print_period_ms) {
        return;
    }
    g_dbg.last_print_ms = now;

    uint16_t p2p_raw_all[USB_DBG_NUM_ADC][USB_DBG_NUM_CH];
    compute_p2p_raw_adc4ch(adc1, num_samples, p2p_raw_all[0]);
    compute_p2p_raw_adc4ch(adc2, num_samples, p2p_raw_all[1]);
    compute_p2p_raw_adc4ch(adc3, num_samples, p2p_raw_all[2]);
    compute_p2p_raw_adc4ch(adc4, num_samples, p2p_raw_all[3]);

    // start/restart accumulation
    if (g_dbg.count == 0u) {
        memset(g_dbg.sum_p2p_raw, 0, sizeof(g_dbg.sum_p2p_raw));
    }

    for (uint32_t a = 0; a < USB_DBG_NUM_ADC; a++) {
        for (uint32_t ch = 0; ch < USB_DBG_NUM_CH; ch++) {
            g_dbg.sum_p2p_raw[a][ch] += p2p_raw_all[a][ch];
        }
    }

    if (g_dbg.count < g_dbg.avg_window) {
        g_dbg.count++;
    }

    // Build one report (bounded, robust)
    char out[1200];
    int n = 0;

#define APPEND(fmt, ...) do { \
    int _r = snprintf(out + n, sizeof(out) - (size_t)n, (fmt), ##__VA_ARGS__); \
    if (_r < 0) goto send; \
    if ((size_t)_r >= (sizeof(out) - (size_t)n)) { \
        n = (int)sizeof(out) - 1; \
        out[n] = '\0'; \
        goto send; \
    } \
    n += _r; \
} while(0)

    APPEND("\r\n=== P2P num_samples=%lu | avg=%lu/%lu ===\r\n",
           (unsigned long)num_samples,
           (unsigned long)g_dbg.count,
           (unsigned long)g_dbg.avg_window);

    for (uint32_t a = 0; a < USB_DBG_NUM_ADC; a++) {
        APPEND("ADC%lu: ", (unsigned long)(a + 1u));

        for (uint32_t ch = 0; ch < USB_DBG_NUM_CH; ch++) {
            uint16_t inst_raw = p2p_raw_all[a][ch];
            uint16_t avg_raw  = (uint16_t)(g_dbg.sum_p2p_raw[a][ch] / g_dbg.count);

            // Convert to mV (integer math, rounded)
            uint32_t inst_mV = (uint32_t)((inst_raw * USB_DBG_ADC_VREF_MV + (USB_DBG_ADC_MAX_COUNTS/2u)) / USB_DBG_ADC_MAX_COUNTS);
            uint32_t avg_mV  = (uint32_t)((avg_raw  * USB_DBG_ADC_VREF_MV + (USB_DBG_ADC_MAX_COUNTS/2u)) / USB_DBG_ADC_MAX_COUNTS);

            APPEND("CH%lu %4u(%lumV) avg %4u(%lumV)  ",
                   (unsigned long)ch,
                   (unsigned)inst_raw, (unsigned long)inst_mV,
                   (unsigned)avg_raw,  (unsigned long)avg_mV);
        }
        APPEND("\r\n");
    }

send:
    if (n < 0) {
        return;
    }
    if (n > (int)sizeof(out)) {
        n = (int)sizeof(out);
    }
    usb_cdc_write_chunked_len((const uint8_t*)out, (uint32_t)n);

    // Reset when the averaging window completes
    if (g_dbg.count >= g_dbg.avg_window) {
        g_dbg.count = 0u;
    }

#undef APPEND
}

void usb_dbg_stream_mic_window_csv(uint8_t mic_index,
                                   const uint16_t *adc_buf,
                                   uint8_t adc_channel,
                                   uint32_t num_samples,
                                   uint32_t sample_rate_hz)
{
    char out[128];
    size_t used = 0u;

    if (!adc_buf || adc_channel >= USB_DBG_NUM_CH) {
        return;
    }

    int n = snprintf(out,
                     sizeof(out),
                     "PASS,mic=%u,rate=%lu,samples=%lu,raw=",
                     (unsigned)mic_index,
                     (unsigned long)sample_rate_hz,
                     (unsigned long)num_samples);
    if (n <= 0) {
        return;
    }

    used = (size_t)n;
    usb_dbg_flush_string(out, &used);

    for (uint32_t i = 0; i < num_samples; i++) {
        uint16_t sample = adc_buf[i * USB_DBG_NUM_CH + adc_channel];
        n = snprintf(out + used,
                     sizeof(out) - used,
                     "%u%s",
                     (unsigned)sample,
                     (i + 1u < num_samples) ? "," : "\r\n");
        if (n < 0) {
            return;
        }

        if ((size_t)n >= (sizeof(out) - used)) {
            usb_dbg_flush_string(out, &used);
            n = snprintf(out,
                         sizeof(out),
                         "%u%s",
                         (unsigned)sample,
                         (i + 1u < num_samples) ? "," : "\r\n");
            if (n < 0) {
                return;
            }
            used = (size_t)n;
            continue;
        }

        used += (size_t)n;
    }

    usb_dbg_flush_string(out, &used);
}

/* ============================================================================
 * STATIC FUNCTIONS
 * ============================================================================ */

static void usb_cdc_write_chunked_len(const uint8_t *data, uint32_t len)
{
    uint32_t off = 0;

    while (off < len) {
        uint16_t chunk = (uint16_t)((len - off) > USB_DBG_CDC_CHUNK ? USB_DBG_CDC_CHUNK : (len - off));

        // Wait until endpoint is ready (robust debug behavior)
        while (CDC_Transmit_FS((uint8_t*)(data + off), chunk) != USBD_OK) {
            HAL_Delay(1);
        }

        off += chunk;
    }
}

static void compute_p2p_raw_adc4ch(const uint16_t *adc_buf,
                                  uint32_t num_samples,
                                  uint16_t p2p_out_raw[USB_DBG_NUM_CH])
{
    uint16_t minv[USB_DBG_NUM_CH] = {4095, 4095, 4095, 4095};
    uint16_t maxv[USB_DBG_NUM_CH] = {0, 0, 0, 0};

    for (uint32_t i = 0; i < num_samples; i++) {
        uint32_t base = i * USB_DBG_NUM_CH;
        for (uint32_t ch = 0; ch < USB_DBG_NUM_CH; ch++) {
            uint16_t v = adc_buf[base + ch];
            if (v < minv[ch]) minv[ch] = v;
            if (v > maxv[ch]) maxv[ch] = v;
        }
    }

    for (uint32_t ch = 0; ch < USB_DBG_NUM_CH; ch++) {
        p2p_out_raw[ch] = (uint16_t)(maxv[ch] - minv[ch]);
    }
}

static void usb_dbg_flush_string(char *buffer, size_t *length)
{
    if (*length == 0u) {
        return;
    }

    usb_cdc_write_chunked_len((const uint8_t *)buffer, (uint32_t)(*length));
    *length = 0u;
}
