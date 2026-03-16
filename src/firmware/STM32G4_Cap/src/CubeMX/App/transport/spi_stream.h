#ifndef SPI_STREAM_H
#define SPI_STREAM_H

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * INCLUDES
 * ========================================================================= */

#include <stddef.h>
#include <stdint.h>

#include "app_main.h"

/* =========================================================================
 * DEFINES
 * ========================================================================= */

#define SPI_CHECKSUM_SIZE_BYTES   (sizeof(uint16_t))
#define SPI_FLOAT_SIZE_BYTES      (sizeof(float))

// arm_rfft_fast_f32() with N=FRAME_SIZE produces FRAME_SIZE total floats 
//(DC and Nyquist bin complex components are omitted)
#define SPI_MIC_PAYLOAD_BYTES (FRAME_SIZE * SPI_FLOAT_SIZE_BYTES)

#define SPI_PACKET_SIZE (SPI_FRAME_HEADER_SIZE_BYTES + \
                         SPI_MIC_PAYLOAD_BYTES + SPI_CHECKSUM_SIZE_BYTES)

#define SPI_FRAME_PACKET_SIZE_BYTES (SPI_FRAME_HEADER_SIZE_BYTES + \
                                     N_MICS * SPI_MIC_PAYLOAD_BYTES + \
                                     SPI_CHECKSUM_SIZE_BYTES)

/* =========================================================================
 * TYPE DEFINITIONS
 * ========================================================================= */

typedef struct {
  uint32_t frame_counter;
  uint32_t batch_counter;
} spi_stream_t;

/* =========================================================================
 * FUNCTION PROTOTYPES
 * ========================================================================= */

/**
 * @brief Initialize SPI stream context.
 */
void spi_stream_init(spi_stream_t *s);

uint32_t spi_stream_next_batch(spi_stream_t *s);

#if SPI_MODE == SPI_FULL_FRAME
size_t spi_stream_build_frame_header(
    spi_stream_t *s,
    uint8_t *dst,
    size_t dst_cap,
    uint32_t batch_id,
    uint16_t mic_index,
    uint16_t fft_size,
    uint32_t sample_rate,
    uint16_t flags,
    size_t   payload_len,
    uint16_t battery_millivolts);

size_t spi_stream_append_mic_payload(
    uint8_t *dst,
    size_t dst_cap,
    const float *fft_data,
    uint16_t fft_size);

size_t spi_stream_finalize_frame(
    uint8_t *dst,
    size_t dst_cap,
    size_t used_len);

#elif SPI_MODE == SPI_SINGLE_MIC
size_t spi_stream_build_mic_packet(
    spi_stream_t *s,
    uint8_t *dst,
    size_t dst_cap,
    uint32_t batch_id,
    uint8_t mic_index,
    const float *fft_bins,
    uint16_t fft_size,
    uint32_t sample_rate,
    uint16_t flags,
    uint16_t battery_millivolts);
#endif

/**
 * @brief Blocking transmit over SPI (hspi4).
 */
void spi_stream_tx_blocking(const uint8_t *buf, size_t len);

void transmit_spi_packet(uint8_t *packet_buffer, uint32_t packet_size);

#ifdef __cplusplus
}
#endif

#endif /* SPI_STREAM_H */