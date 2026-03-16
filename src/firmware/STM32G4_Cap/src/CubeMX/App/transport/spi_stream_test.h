#ifndef SPI_STREAM_TEST_H
#define SPI_STREAM_TEST_H

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

void spi_stream_unit_test_build_packet(void);
void spi_stream_unit_test_nulls(void);
void spi_stream_unit_test_small_cap(void);
void spi_stream_unit_test_frame_counter(void);
void spi_loopback_unit_test(void);
void spi_loopback_unit_test2(void);

#ifdef __cplusplus
}
#endif

#endif /* SPI_STREAM_TEST_H */