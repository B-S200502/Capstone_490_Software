/**
 * @file spi_stream_test.c
 * @brief Unit tests for SPI stream packet builder
 * @details 
 */

/* =========================================================================
 * INCLUDES
 * ========================================================================= */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "protocol/spi_protocol.h"

#include "spi.h"
#include "spi_stream_test.h"
#include "spi_stream.h"
#include "spi_dma.h"

#include "usb/usb_debug.h"

#include "app_main.h"

/* =========================================================================
 * DEFINES & CONSTANTS
 * ========================================================================= */

/* =========================================================================
 * STATIC VARIABLES
 * ========================================================================= */

//  extern volatile uint32_t spi4_tx_dma_irq_count;
//  extern volatile uint32_t spi4_rx_dma_irq_count;    
/* =========================================================================
 * FORWARD DECLARATIONS
 * ========================================================================= */
static int buffers_equal(const uint8_t *a, const uint8_t *b, size_t n, 
                         size_t *first_bad);
/* static void module_init(void); */
/* static void module_process(void); */

/* =========================================================================
 * PUBLIC FUNCTIONS
 * ========================================================================= */
/**
 * @brief Dumps a byte buffer in hex format over USB CDC for debugging.
 * @param[in] p Pointer to the byte buffer to dump
 * @param[in] n Number of bytes in the buffer
 * @return void
 */
static void dump_hex(const uint8_t *p, size_t n)
{
  for (size_t i = 0; i < n; i++) {
      usb_printf("%02X%s", p[i], ((i + 1) % 16 == 0) ? "\r\n" : " ");
  }
  if (n % 16 != 0) usb_printf("\r\n");
}

/**
 * @brief Safe float read from possibly unaligned payload
 * @param[in] p Pointer to 4 bytes representing a float in IEEE-754 format
 * @return The float value read from the byte array
 */ 
static float read_f32_le(const uint8_t *p)
{
  float f;
  memcpy(&f, p, sizeof(f));
  return f;
}

// Recompute checksum exactly like your function
static uint16_t checksum16_sum_bytes_ref(const uint8_t *data, size_t length)
{
  uint16_t sum = 0;
  for (size_t i = 0; i < length; i++) {
      sum = (uint16_t)(sum + data[i]);
  }
  return sum;
}

#if SPI_MODE == SPI_SINGLE_MIC
void spi_stream_unit_test_build_packet(void)
{
    usb_printf("\r\n=== spi_stream_build_mic_packet UNIT TEST ===\r\n");

  // Test parameters
    const uint32_t batch_id    = 77;
    const uint16_t mic_index   = 3;
  const uint16_t fft_size    = 16;
  const uint32_t sample_rate = 100000;
  const uint16_t bin_count   = 8; // small for easy printing

  // Build a known complex payload: [Re0, Im0, Re1, Im1, ...]
  // Your builder uses payload_len = 2 * bin_count * sizeof(float)
  float fft_bins[2 * bin_count];
  for (uint16_t k = 0; k < bin_count; k++) {
      fft_bins[2*k + 0] = (float)k + 0.25f;   // Re
      fft_bins[2*k + 1] = -(float)k - 0.75f;  // Im
  }

  spi_stream_t stream;
  spi_stream_init(&stream);

  // Ensure dst is aligned to 4 bytes to avoid unaligned header stores
  union {
      uint32_t align;
      uint8_t  bytes[256];
  } pkt;

  memset(pkt.bytes, 0xEE, sizeof(pkt.bytes)); // poison to detect what gets written

  const size_t total = spi_stream_build_mic_packet(
      &stream,
      pkt.bytes,
      sizeof(pkt.bytes),
      batch_id,
      mic_index,
      fft_bins,
      fft_size,
      sample_rate,
      SPI_FRAME_FLAG_PAYLOAD_COMPLEX,
      0u);

  usb_printf("build returned total_len=%u\r\n", (unsigned)total);

  if (total == 0) {
      usb_printf("FAIL: build returned 0\r\n");
      return;
  }

  // 1) Check header values (read via memcpy to avoid alignment assumptions)
  SPI_FrameHeader_t hdr;
  memcpy(&hdr, pkt.bytes, sizeof(hdr));

  usb_printf("Header:\r\n");
  usb_printf("  magic       = 0x%08lX (expect 0x%08lX)\r\n",
             (unsigned long)hdr.magic, (unsigned long)SPI_MAGIC);
  usb_printf("  version      = %u (expect %u)\r\n",
             (unsigned)hdr.version, (unsigned)SPI_VERSION);
  usb_printf("  header_len   = %u (expect %u)\r\n",
             (unsigned)hdr.header_len, (unsigned)sizeof(SPI_FrameHeader_t));
  usb_printf("  batch_id     = %lu (expect %lu)\r\n",
             (unsigned long)hdr.batch_id, (unsigned long)batch_id);
  usb_printf("  mic_index    = %u (expect %u)\r\n",
              (unsigned)hdr.mic_index, (unsigned)mic_index);
  usb_printf("  fft_size     = %u (expect %u)\r\n",
             (unsigned)hdr.fft_size, (unsigned)fft_size);
  usb_printf("  sample_rate  = %lu (expect %lu)\r\n",
             (unsigned long)hdr.sample_rate, (unsigned long)sample_rate);
  usb_printf("  flags        = 0x%04X (expect 0x%04X)\r\n",
             (unsigned)hdr.flags, (unsigned)SPI_FRAME_FLAG_PAYLOAD_COMPLEX);
  usb_printf("  battery_mv   = %u (expect 0)\r\n",
             (unsigned)hdr.battery_mv);
  usb_printf("  payload_len  = %lu (expect %lu)\r\n",
             (unsigned long)hdr.payload_len,
             (unsigned long)(2u * (unsigned long)bin_count * (unsigned long)sizeof(float)));

  // 2) Verify payload floats match what we put in
  const size_t header_len  = sizeof(SPI_FrameHeader_t);
  const size_t payload_len = 2u * (size_t)bin_count * sizeof(float);
  const uint8_t *payload   = pkt.bytes + header_len;

  usb_printf("Payload check (first %u complex bins):\r\n", (unsigned)bin_count);

  int payload_ok = 1;
  for (uint16_t k = 0; k < bin_count; k++) {

    // Read back from packet payload (byte-safe)
    float re = read_f32_le(payload + (size_t)(2u*k + 0u) * sizeof(float));
    float im = read_f32_le(payload + (size_t)(2u*k + 1u) * sizeof(float));

    // Expected from source array
    float re_exp = fft_bins[2u*k + 0u];
    float im_exp = fft_bins[2u*k + 1u];

    // Convert floats to raw IEEE-754 bit patterns for printing
    uint32_t re_u, im_u, re_exp_u, im_exp_u;
    memcpy(&re_u,     &re,     sizeof(re_u));
    memcpy(&im_u,     &im,     sizeof(im_u));
    memcpy(&re_exp_u, &re_exp, sizeof(re_exp_u));
    memcpy(&im_exp_u, &im_exp, sizeof(im_exp_u));

    // Print as hex
    usb_printf("  k=%u: re=0x%08lX (exp 0x%08lX), im=0x%08lX (exp 0x%08lX)%s\r\n",
                (unsigned)k,
                (unsigned long)re_u,     (unsigned long)re_exp_u,
                (unsigned long)im_u,     (unsigned long)im_exp_u,
                (re_u == re_exp_u && im_u == im_exp_u) ? "" : "  <-- MISMATCH");

    // Bitwise compare (exact)
    if (re_u != re_exp_u || im_u != im_exp_u) {
        payload_ok = 0;
    }
  }
  usb_printf("Payload: %s\r\n", payload_ok ? "PASS" : "FAIL");

  // 3) Verify checksum stored matches recomputed
  const uint8_t *cs_ptr = payload + payload_len;
  uint16_t cs_stored;
  memcpy(&cs_stored, cs_ptr, sizeof(cs_stored));

  const uint16_t cs_calc = checksum16_sum_bytes_ref(pkt.bytes, header_len + payload_len);

  usb_printf("Checksum:\r\n");
  usb_printf("  stored = 0x%04X\r\n", cs_stored);
  usb_printf("  calc   = 0x%04X\r\n", cs_calc);
  usb_printf("Checksum: %s\r\n", (cs_stored == cs_calc) ? "PASS" : "FAIL");

  // 4) Sanity: total length expected
  const size_t expected_total = header_len + payload_len + sizeof(uint16_t);
  usb_printf("Total length: %u (expected %u) => %s\r\n",
              (unsigned)total, (unsigned)expected_total,
              (total == expected_total) ? "PASS" : "FAIL");

  // 5) Hex dump of header + a bit of payload
  usb_printf("Hex dump (first 64 bytes):\r\n");
  dump_hex(pkt.bytes, (total < 64) ? total : 64);

  usb_printf("=== END UNIT TEST ===\r\n");
}


void spi_stream_unit_test_nulls(void)
{
  spi_stream_t s;
  spi_stream_init(&s);

  uint8_t dst[64];
  float bins[16];

  usb_printf("\r\n=== NULL TESTS ===\r\n");

  usb_printf("null stream => %u\r\n",
      (unsigned)spi_stream_build_mic_packet(NULL, dst, sizeof(dst), 0u, 0u, bins, 1024, 100000, 0u, 0u));

  usb_printf("null dst => %u\r\n",
      (unsigned)spi_stream_build_mic_packet(&s, NULL, sizeof(dst), 0u, 0u, bins, 1024, 100000, 0u, 0u));

  usb_printf("null bins => %u\r\n",
      (unsigned)spi_stream_build_mic_packet(&s, dst, sizeof(dst), 0u, 0u, NULL, 1024, 100000, 0u, 0u));
}

void spi_stream_unit_test_small_cap(void)
{
  spi_stream_t s;
  spi_stream_init(&s);

  uint8_t dst[32];     // intentionally too small
  float bins[16];      // 2*8 floats

  usb_printf("\r\n=== CAPACITY TEST ===\r\n");
    size_t n = spi_stream_build_mic_packet(&s, dst, sizeof(dst), 0u, 0u, bins, 1024, 100000, 0u, 0u);
  usb_printf("dst_cap=%u => returned %u (expect 0)\r\n",
              (unsigned)sizeof(dst), (unsigned)n);
}

void spi_stream_unit_test_frame_counter(void)
{
  spi_stream_t s;
  spi_stream_init(&s);

  union { uint32_t a; uint8_t b[256]; } pkt;
  float bins[16];

  usb_printf("\r\n=== FRAME COUNTER TEST ===\r\n");
  for (int i = 0; i < 3; i++) {
            size_t n = spi_stream_build_mic_packet(&s, pkt.b, sizeof(pkt.b), 55u, 0u, bins, 16, 100000, 0u, 0u);
      SPI_FrameHeader_t hdr;
      memcpy(&hdr, pkt.b, sizeof(hdr));
            usb_printf("call %d: len=%u, frame_counter=%u, batch_id=%u\r\n",
                       i, (unsigned)n, hdr.frame_counter, (unsigned)hdr.batch_id);
                       
  }
}

void spi_loopback_unit_test(void)
{
    usb_printf("\r\n=== SPI4 LOOPBACK UNIT TEST (MOSI->MISO) ===\r\n");

    // ---------------------------------------------------------------------
    // 1) Build a small packet into tx[]
    // ---------------------------------------------------------------------
    const uint32_t batch_id    = 2;
    const uint16_t mic_index   = 1;
    const uint16_t fft_size    = 512;
    const uint32_t sample_rate = 48000;
    const uint16_t bin_count   = 256; // keep small to start

    const size_t transfer_size = SPI_PACKET_SIZE; // intentionally exact size for header+payload+checksum

    float fft_bins[2 * bin_count];
    for (uint16_t k = 0; k < bin_count; k++) {
        fft_bins[2u*k + 0u] = (float)k + 0.25f;
        fft_bins[2u*k + 1u] = -(float)k - 0.75f;
    }

    spi_stream_t stream;
    spi_stream_init(&stream);

    // Align buffers (DMA/HAL generally doesn't require for blocking, but it's nice)
    uint8_t *tx_buf = spi_dma_get_tx_buffer();
    uint8_t *rx_buf = spi_dma_get_rx_buffer();
    memset(tx_buf, 0x00, transfer_size);
    memset(rx_buf, 0xCC, transfer_size);

    const size_t tx_len = spi_stream_build_mic_packet(
        &stream,
        tx_buf,
        transfer_size,
        batch_id,
        mic_index,
        fft_bins,
        fft_size,
        sample_rate,
        SPI_FRAME_FLAG_PAYLOAD_COMPLEX,
        0u);

    if (tx_len == 0) {
        usb_printf("FAIL: spi_stream_build_mic_packet returned 0\r\n");
        return;
    }

    usb_printf("Built packet len=%u\r\n", (unsigned)tx_len);
    usb_printf("TX (first 64 bytes):\r\n");
    dump_hex(tx_buf, (tx_len < 64) ? tx_len : 64);

    // ---------------------------------------------------------------------
    // 2) SPI transmit+receive
    // ---------------------------------------------------------------------
    int status = spi_stream_txrx_dma(tx_buf, rx_buf, (uint16_t)tx_len);

    usb_printf("spi_stream_txrx_dma returned %d\r\n", status);

    while(!get_spi_dma_done()) {
        usb_printf("Waiting for SPI DMA to complete...\r\n");
        HAL_Delay(10);
    };

    usb_printf("RX (first 64 bytes):\r\n");
    dump_hex(rx_buf, (tx_len < 64) ? tx_len : 64);

    // ---------------------------------------------------------------------
    // 3) Compare
    // ---------------------------------------------------------------------
    size_t first_bad = 0;
    int ok = buffers_equal(tx_buf, rx_buf, tx_len, &first_bad);

    if (!ok) {
        usb_printf("FAIL: TX != RX at byte index %u\r\n", (unsigned)first_bad);
        usb_printf("TX around mismatch:\r\n");
        size_t start = (first_bad > 16) ? (first_bad - 16) : 0;
        size_t end   = (first_bad + 16 < tx_len) ? (first_bad + 16) : tx_len;
        dump_hex(&tx_buf[start], end - start);

        usb_printf("RX around mismatch:\r\n");
        dump_hex(&rx_buf[start], end - start);
        return;
    }

    usb_printf("PASS: TX == RX (%u bytes)\r\n", (unsigned)tx_len);

    // ---------------------------------------------------------------------
    // 4) Optional: parse header from RX and sanity check
    // ---------------------------------------------------------------------
    SPI_FrameHeader_t hdr;
    memcpy(&hdr, rx_buf, sizeof(hdr));
    usb_printf("RX header magic=0x%08lX, batch_id=%lu, payload_len=%lu\r\n",
               (unsigned long)hdr.magic,
               (unsigned long)hdr.batch_id,
               (unsigned long)hdr.payload_len);

    usb_printf("=== END LOOPBACK TEST ===\r\n");
}
#endif

void spi_loopback_unit_test2(void)
{
  usb_printf("\r\n=== SPI4 LOOPBACK UNIT TEST (16-MIC FRAME) ===\r\n");

  const uint32_t batch_id    = 2;
  const uint16_t mic_index   = N_MICS;   // protocol sentinel for multi-mic frame
  const uint16_t fft_size    = 32;
  const uint32_t sample_rate = 48000;
  const uint16_t flags       = SPI_FRAME_FLAG_PAYLOAD_COMPLEX;
  const uint16_t battery_mv  = 0u;

  const size_t payload_len   = N_MICS * fft_size * sizeof(float);

  size_t offset    = 0u;
  size_t final_len = 0u;

  spi_stream_t stream;
  spi_stream_init(&stream);

  uint8_t *tx_buf = spi_dma_get_tx_buffer();
  uint8_t *rx_buf = spi_dma_get_rx_buffer();

  memset(tx_buf, 0x00, SPI_FRAME_PACKET_SIZE_BYTES);
  memset(rx_buf, 0xCC, 4096);

  /* ------------------------------------------------------------------ */
  /* 1) Build one full multi-mic frame into tx_buf                      */
  /* ------------------------------------------------------------------ */
  offset = spi_stream_build_frame_header(&stream,
                                         tx_buf,
                                         SPI_FRAME_PACKET_SIZE_BYTES,
                                         batch_id,
                                         mic_index,
                                         fft_size,
                                         sample_rate,
                                         flags,
                                         payload_len,
                                         battery_mv);

  if (offset == 0u) {
    usb_printf("FAIL: spi_stream_build_frame_header returned 0\r\n");
    return;
  }

  /* Build synthetic FFT payloads for mic0..mic15 */
  for (uint16_t mic = 0; mic < N_MICS; mic++) {
    float mic_fft_buffer[fft_size];

    for (uint16_t i = 0; i < fft_size; i++) {
      /* Easy-to-recognize per-mic pattern */
      mic_fft_buffer[i] = (float)(mic * 1000u + i) + 0.25f;
    }

    size_t appended_len = spi_stream_append_mic_payload(
        tx_buf + offset,
        SPI_FRAME_PACKET_SIZE_BYTES - offset,
        mic_fft_buffer,
        fft_size);

    if (appended_len == 0u) {
      usb_printf("FAIL: spi_stream_append_mic_payload returned 0 for mic %u\r\n",
                 (unsigned)mic);
      return;
    }

    offset += appended_len;
  }

  final_len = spi_stream_finalize_frame(tx_buf,
                                        SPI_FRAME_PACKET_SIZE_BYTES - offset,
                                        offset);

  if (final_len == 0u) {
    usb_printf("FAIL: spi_stream_finalize_frame returned 0\r\n");
    return;
  }

  usb_printf("Built multi-mic frame len=%u\r\n", (unsigned)final_len);
  usb_printf("TX (first 64 bytes):\r\n");
  dump_hex(tx_buf, (final_len < 64u) ? final_len : 64u);

  /* ------------------------------------------------------------------ */
  /* 2) SPI transmit+receive                                            */
  /* ------------------------------------------------------------------ */
  clear_spi_dma_done();

  int status = spi_stream_txrx_dma(tx_buf, rx_buf, (uint16_t)final_len);
  usb_printf("spi_stream_txrx_dma returned %d\r\n", status);

  while (!get_spi_dma_done()) {
    usb_printf("Waiting for SPI DMA to complete...\r\n");
    HAL_Delay(10);
  }

  usb_printf("RX (first 64 bytes):\r\n");
  dump_hex(rx_buf, (final_len < 64u) ? final_len : 64u);

  /* ------------------------------------------------------------------ */
  /* 3) Compare TX and RX                                               */
  /* ------------------------------------------------------------------ */
  size_t first_bad = 0u;
  int ok = buffers_equal(tx_buf, rx_buf, final_len, &first_bad);

  if (!ok) {
    usb_printf("FAIL: TX != RX at byte index %u\r\n", (unsigned)first_bad);

    usb_printf("TX around mismatch:\r\n");
    size_t start = (first_bad > 16u) ? (first_bad - 16u) : 0u;
    size_t end   = (first_bad + 16u < final_len) ? (first_bad + 16u) : final_len;
    dump_hex(&tx_buf[start], end - start);

    usb_printf("RX around mismatch:\r\n");
    dump_hex(&rx_buf[start], end - start);
    return;
  }

  usb_printf("PASS: TX == RX (%u bytes)\r\n", (unsigned)final_len);

  /* ------------------------------------------------------------------ */
  /* 4) Parse header from RX and sanity check                           */
  /* ------------------------------------------------------------------ */
  SPI_FrameHeader_t hdr;
  memcpy(&hdr, rx_buf, sizeof(hdr));

  usb_printf("RX header magic=0x%08lX, batch_id=%lu, mic_index=%u, payload_len=%lu\r\n",
             (unsigned long)hdr.magic,
             (unsigned long)hdr.batch_id,
             (unsigned)hdr.mic_index,
             (unsigned long)hdr.payload_len);

  /* ------------------------------------------------------------------ */
  /* 5) Optional payload sanity check                                   */
  /* ------------------------------------------------------------------ */
  const uint8_t *rx_payload = rx_buf + sizeof(SPI_FrameHeader_t);
  size_t mic_stride_bytes = fft_size * sizeof(float);

  float mic0_bin0  = read_f32_le(rx_payload + 0u  * mic_stride_bytes);
  float mic1_bin0  = read_f32_le(rx_payload + 1u  * mic_stride_bytes);
  float mic15_bin0 = read_f32_le(rx_payload + 15u * mic_stride_bytes);

  usb_printf("Payload spot-check:\r\n");

  usb_printf("mic0[0]   = %d.%02d (expected 0.25)\r\n",
            PRINT_F2(mic0_bin0));

  usb_printf("mic1[0]   = %d.%02d (expected 1000.25)\r\n",
            PRINT_F2(mic1_bin0));

  usb_printf("mic15[0]  = %d.%02d (expected 15000.25)\r\n",
            PRINT_F2(mic15_bin0));

  usb_printf("=== END LOOPBACK TEST ===\r\n");
}
/* ============================================================================
 * STATIC FUNCTIONS
 * ============================================================================ */

/**
 * @brief Checks TX and RX buffers for equality
 * @return 1 if equal, 0 if not.
 */
static int buffers_equal(const uint8_t *a, const uint8_t *b, size_t n, size_t *first_bad)
{
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            if (first_bad) *first_bad = i;
            return 0;
        }
    }
    return 1;
}

