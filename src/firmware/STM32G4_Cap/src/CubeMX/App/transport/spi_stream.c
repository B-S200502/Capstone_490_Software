/**
 * @file spi_stream.c
 * @brief SPI streaming module for transmitting FFT data over SPI.
 * @details 
 */

/* =========================================================================
 * INCLUDES
 * ========================================================================= */
#include "spi.h"
#include "spi_stream.h"
#include "../protocol/spi_protocol.h"
#include "app_main.h"

#include <string.h>

/* =========================================================================
 * DEFINES & CONSTANTS
 * ========================================================================= */


/* =========================================================================
 * STATIC VARIABLES
 * ========================================================================= */

/* =========================================================================
 * FORWARD DECLARATIONS
 * ========================================================================= */

 /**
 * @brief Calculates a simple 16-bit checksum by summing all bytes
 * @return The 16-bit checksum value
 */
static uint16_t checksum16_sum_bytes(const uint8_t *data, size_t length);

/**
 * @brief Fills SPI frame header
 * @return void
 */
static void spi_stream_fill_header(SPI_FrameHeader_t *hdr,
                                   spi_stream_t *s,
                                   uint32_t batch_id,
                                   uint16_t mic_index,
                                   uint16_t fft_size,
                                   uint32_t sample_rate,
                                   uint16_t flags,
                                   uint16_t battery_millivolts,
                                   uint32_t payload_len);

/* =========================================================================
 * PUBLIC FUNCTIONS
 * ========================================================================= */

void spi_stream_init(spi_stream_t *s)
{
    if (!s) return;
    s->frame_counter = 0;
    s->batch_counter = 0;
}

uint32_t spi_stream_next_batch(spi_stream_t *s)
{
    if (!s) {
        return 0u;
    }

    return s->batch_counter++;
}

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
    uint16_t battery_millivolts)
{
  if (!s || !dst)
      return 0;

  SPI_FrameHeader_t hdr = {0};

  const size_t header_len = sizeof(hdr);

  const size_t checksum_len = SPI_CHECKSUM_SIZE_BYTES;

  const size_t total_len =
      header_len + payload_len + checksum_len;

  if (dst_cap < total_len)
      return 0;

  /* ===== HEADER ===== */
  spi_stream_fill_header(&hdr,
                          s,
                          batch_id,
                          mic_index,
                          fft_size,
                          sample_rate,
                          flags,
                          battery_millivolts,
                          (uint32_t)payload_len);

  memcpy(dst, &hdr, header_len);
  return header_len;
}

size_t spi_stream_append_mic_payload(
    uint8_t *dst,
    size_t dst_cap,
    const float *fft_data,
    uint16_t fft_size)
{
  if (!dst || !fft_data)
    return 0;

  const size_t payload_bytes = fft_size * sizeof(float);

  if (dst_cap < payload_bytes)
      return 0;

  memcpy(dst, fft_data, payload_bytes);

  return payload_bytes;
}

size_t spi_stream_finalize_frame(
    uint8_t *dst,
    size_t dst_cap,
    size_t used_len)
{
  uint16_t checksum;

  if (!dst)
    return 0;

  if (dst_cap < SPI_CHECKSUM_SIZE_BYTES)
    return 0;

  checksum = checksum16_sum_bytes(dst, used_len);

  memcpy(dst + used_len, &checksum, sizeof(checksum));

  return used_len + SPI_CHECKSUM_SIZE_BYTES;
}

#elif SPI_MODE == SPI_SINGLE_MIC
size_t spi_stream_build_mic_packet(
    spi_stream_t *s,
    uint8_t *dst,
    size_t dst_cap,
    uint32_t batch_id,
    uint16_t mic_index,
    const float *fft_data,
    uint16_t fft_size,
    uint32_t sample_rate,
    uint16_t flags,
    uint16_t battery_millivolts)
{
    if (!s || !dst || !fft_data)
        return 0;

    SPI_FrameHeader_t hdr = {0};

    const size_t header_len = sizeof(hdr);

    // Payload length is the same as fft_size * sizeof(float)
    const size_t payload_len =
        fft_size * sizeof(float);

    const size_t checksum_len = sizeof(uint16_t);

    const size_t total_len =
        header_len + payload_len + checksum_len;

    if (dst_cap < total_len)
        return 0;

    /* ===== HEADER ===== */

    spi_stream_fill_header(&hdr,
                           s,
                           batch_id,
                           mic_index,
                           fft_size,
                           sample_rate,
                           flags,
                           battery_millivolts,
                           (uint32_t)payload_len);

    memcpy(dst, &hdr, header_len);

    /* ===== PAYLOAD ===== */

    uint8_t *payload_ptr = dst + header_len;

    memcpy(payload_ptr, fft_data, payload_len);

    /* ===== CHECKSUM ===== */

    uint16_t checksum =
        checksum16_sum_bytes(dst, header_len + payload_len);

    memcpy(payload_ptr + payload_len,
           &checksum,
           sizeof(checksum));

    return total_len;
}
#endif

void spi_stream_tx_blocking(const uint8_t *buf, size_t len)
{
    if (!buf || len == 0) return;

    // HAL takes uint16_t length; enforce that here
    if (len > 0xFFFFu) return;

    (void)HAL_SPI_Transmit(&hspi4, (uint8_t *)buf, (uint16_t)len, HAL_MAX_DELAY);
}

/**
 * @brief Public function description
 * @param[in] param1 Description of input parameter
 * @return Description of return value
 */

/* ============================================================================
 * STATIC FUNCTIONS
 * ============================================================================ */

static uint16_t checksum16_sum_bytes(const uint8_t *data, size_t length)
{
    uint16_t sum = 0;
    for (size_t i = 0; i < length; i++) {
        sum = (uint16_t)(sum + data[i]);
    }
    return sum;
}

static void spi_stream_fill_header(SPI_FrameHeader_t *hdr,
                                   spi_stream_t *s,
                                   uint32_t batch_id,
                                   uint16_t mic_index,
                                   uint16_t fft_size,
                                   uint32_t sample_rate,
                                   uint16_t flags,
                                   uint16_t battery_millivolts,
                                   uint32_t payload_len)
{
    hdr->magic = SPI_MAGIC;
    hdr->version = (uint16_t)SPI_VERSION;
    hdr->header_len = (uint16_t)sizeof(*hdr);
    hdr->frame_counter = s->frame_counter++;
    hdr->batch_id = batch_id;
    hdr->mic_index = mic_index;
    hdr->fft_size = fft_size;
    hdr->sample_rate = sample_rate;
    hdr->flags = flags;
    hdr->payload_len = payload_len;
    hdr->battery_mv = battery_millivolts;
    hdr->clipping_flag = (uint8_t)((flags & SPI_FRAME_FLAG_TIME_CLIPPING) ? 1u : 0u);
    hdr->reserved0 = 0u;
    hdr->reserved1 = 0u;
}
/**
 * @brief Static helper function description
 * @return void
 */


