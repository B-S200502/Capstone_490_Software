#ifndef TEMPLATE_H
#define TEMPLATE_H

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

/* =========================================================================
 * FUNCTION PROTOTYPES
 * ========================================================================= */

uint8_t *spi_dma_get_tx_buffer(void);
uint8_t *spi_dma_get_rx_buffer(void);
uint8_t get_spi_dma_busy(void);
uint8_t get_spi_dma_done(void);

void clear_spi_dma_done(void);

int spi_stream_tx_dma(uint8_t *tx_buf, uint16_t len);
int spi_stream_txrx_dma(uint8_t *tx_buf, uint8_t *rx_buf, uint16_t len);


#ifdef __cplusplus
}
#endif

#endif /* TEMPLATE_H */