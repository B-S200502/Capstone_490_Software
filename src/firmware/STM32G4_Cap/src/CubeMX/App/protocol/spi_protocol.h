#ifndef __SPI_PROTOCOL_H__
#define __SPI_PROTOCOL_H__

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * INCLUDES
 * ========================================================================= */

#include <stdint.h>
#include "app_main.h"

/* =========================================================================
 * DEFINES
 * ========================================================================= */

#define SPI_MAGIC 0xAABBCCDDu
#define SPI_VERSION 1u

#define SPI_FRAME_HEADER_SIZE_BYTES (sizeof(SPI_FrameHeader_t))

#define SPI_FRAME_FLAG_SYNCED_ALL_MICS   (1u << 0)
#define SPI_FRAME_FLAG_PAYLOAD_COMPLEX   (1u << 1)
#define SPI_FRAME_FLAG_PAYLOAD_TIME      (1u << 2)
#define SPI_FRAME_FLAG_BATTERY_VALID     (1u << 3)
#define SPI_FRAME_FLAG_SECOND_HALF       (1u << 4)
#define SPI_FRAME_FLAG_TIME_CLIPPING     (1u << 5)

/* =========================================================================
 * TYPE DEFINITIONS
 * ========================================================================= */

typedef struct __attribute__((packed)) {
    uint32_t magic;        // Frame magic (e.g., 0xAABBCCDD)
    uint16_t version;      // Protocol version
    uint16_t header_len;   // sizeof(SPI_FrameHeader_t)
    uint32_t frame_counter; // Incremented for each mic packet sent (wraps on uint32_t)
    uint16_t batch_id;     // Shared by all microphones from one synchronized sweep
    uint8_t  mic_index;    // Microphone index for this payload
    uint16_t fft_size;     // FFT size (e.g., 1024)
    uint32_t sample_rate;  // Sampling rate (Hz)
    uint16_t flags;        // Payload and sync flags
    uint16_t payload_len;  // Bytes following header (FFT payload size)
    uint16_t battery_mv;   // Battery sense value in millivolts after divider compensation
    uint8_t  clipping_flag; // 1 if time-domain clipping detected in this window, else 0
    uint16_t reserved0;    // Future use
    uint16_t reserved1;    // Future use
} SPI_FrameHeader_t;

/* =========================================================================
 * FUNCTION PROTOTYPES
 * ========================================================================= */

#ifdef __cplusplus
static_assert(sizeof(SPI_FrameHeader_t) == SPI_FRAME_HEADER_SIZE_BYTES,
              "SPI_FrameHeader_t size mismatch");
#else
_Static_assert(sizeof(SPI_FrameHeader_t) == SPI_FRAME_HEADER_SIZE_BYTES,
               "SPI_FrameHeader_t size mismatch");
#endif
#ifdef __cplusplus
}
#endif

#endif /* __SPI_PROTOCOL_H__ */