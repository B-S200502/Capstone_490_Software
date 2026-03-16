/**
 * @file spi_dma.c
 * @brief Brief description of module functionality
 * @details Detailed description of what this module does and its responsibilities
 */

/* =========================================================================
 * INCLUDES
 * ========================================================================= */

#include <stdint.h>
#include <stddef.h>

#include "spi_dma.h"
#include "protocol/spi_protocol.h"
#include "metrics/timing.h"
#include "spi_stream.h"
#include "spi.h"

#include "dma.h"

#include "usb/usb_debug.h"

/* =========================================================================
 * DEFINES & CONSTANTS
 * ========================================================================= */

/* =========================================================================
 * STATIC VARIABLES
 * ========================================================================= */

#if SPI_MODE == SPI_FULL_FRAME
static uint8_t spi_tx[SPI_FRAME_PACKET_SIZE_BYTES];
static uint8_t spi_rx[4096];
#else
static uint8_t spi_tx[2][SPI_PACKET_SIZE];
static uint8_t spi_rx[2][SPI_PACKET_SIZE];
#endif

volatile uint8_t spi_dma_busy = 0;
volatile uint8_t spi_dma_done = 0;
static volatile uint8_t spi_dma_error = 0;

static volatile uint8_t spi_active_buf = 0;   // buffer currently in DMA
static volatile uint8_t spi_fill_buf   = 1;   // buffer app may fill
static volatile uint16_t spi_dma_len  = 0;

/* =========================================================================
 * FORWARD DECLARATIONS
 * ========================================================================= */

/* static void module_init(void); */
/* static void module_process(void); */

/* =========================================================================
 * PUBLIC FUNCTIONS
 * ========================================================================= */

#if SPI_MODE == SPI_FULL_FRAME
uint8_t *spi_dma_get_tx_buffer(void) { return spi_tx; }
uint8_t *spi_dma_get_rx_buffer(void) { return spi_rx; }
#else
uint8_t *spi_dma_get_tx_buffer(void) { return spi_tx[spi_fill_buf]; }
uint8_t *spi_dma_get_rx_buffer(void) { return spi_rx[spi_fill_buf]; }
#endif

uint8_t get_spi_dma_busy(void) { return spi_dma_busy; }
uint8_t get_spi_dma_done(void) { return spi_dma_done; }

void clear_spi_dma_done(void) { spi_dma_done = 0u; }

int spi_stream_tx_dma(uint8_t *tx_buf, uint16_t len)
{
  HAL_StatusTypeDef status;

  if ((tx_buf == NULL) || (len == 0u)) {
      return 0;
  }

  // if (spi_dma_busy) {
  //     return 0;
  // }

  const uint8_t buf_index = spi_fill_buf;
  spi_active_buf = buf_index;
  spi_fill_buf ^= 1u;

  spi_dma_busy = 1u;
  spi_dma_done = 0u;
  spi_dma_error = 0u;
  spi_dma_len = len;

  status = HAL_SPI_Transmit_DMA(&hspi4, tx_buf, len);

  if (status != HAL_OK) {
      spi_dma_busy = 0u;
      return 0;
  }

  /* Set MCU_STATUS (PE0) high after DMA is armed so Pi sees "frame ready" only when we are ready. */
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_0, GPIO_PIN_SET);

  return 1;
}

int spi_stream_txrx_dma(uint8_t *tx_buf, uint8_t *rx_buf, uint16_t len)
{
  HAL_StatusTypeDef status;

  if ((tx_buf == NULL) || (rx_buf == NULL) || (len == 0u)) {
    return 0;
  }

  while (spi_dma_busy) {
    usb_printf("%s", "SPI DMA busy, waiting...\r\n");
    HAL_Delay(10);
  }
  // if (spi_dma_busy) {
  //     return 0;
  // }

  const uint8_t buf_index = spi_fill_buf;
  spi_active_buf = buf_index;
  spi_fill_buf ^= 1u;

  spi_dma_busy  = 1u;
  spi_dma_done  = 0u;
  spi_dma_error = 0u;
  spi_dma_len   = len;

  usb_printf("Starting SPI TxRx DMA: len=%u\r\n", (unsigned)len);

  // Set MCU_STATUS pin to HIGH
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_0, GPIO_PIN_SET);

  status = HAL_SPI_TransmitReceive_DMA(&hspi4, tx_buf, rx_buf, len);

  if (status != HAL_OK) {
    spi_dma_busy = 0u;
    return 0;
  }

  // usb_printf("TX CNDTR=%lu RX CNDTR=%lu\r\n",
  //            (unsigned long)hspi4.hdmatx->Instance->CNDTR,
  //            (unsigned long)hspi4.hdmarx->Instance->CNDTR);

  // HAL_Delay(1000);

  // usb_printf("TX CNDTR=%lu RX CNDTR=%lu\r\n",
  //            (unsigned long)hspi4.hdmatx->Instance->CNDTR,
  //            (unsigned long)hspi4.hdmarx->Instance->CNDTR);

  return 1;
}

 void HAL_SPI_TxRxCpltCallback(SPI_HandleTypeDef *hspi)
{
  /* Ensure this interrupt came from our SPI peripheral */
  if (hspi != &hspi4) {
      return;
  }

  /* DMA transfer completed successfully */
  spi_dma_busy  = 0u;
  spi_dma_done  = 1u;

  // Reset MCU_STATUS pin to LOW
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_0, GPIO_PIN_RESET);
}


void HAL_SPI_TxCpltCallback(SPI_HandleTypeDef *hspi)
{
  /* Only relevant if using TX-only DMA later */
  if (hspi != &hspi4) {
      return;
  }

  // usb_printf("%s", "SPI Tx DMA complete callback\r\n");
  spi_dma_busy  = 0u;
  spi_dma_done  = 1u;

  // Reset MCU_STATUS pin to LOW (PE0)
  HAL_GPIO_WritePin(GPIOE, GPIO_PIN_0, GPIO_PIN_RESET);
  end_performance_measurement();
}


void HAL_SPI_RxCpltCallback(SPI_HandleTypeDef *hspi)
{
  /* Rarely used for SPI streaming, but included for completeness */
  if (hspi != &hspi4) {
      return;
  }

  spi_dma_busy  = 0u;
  spi_dma_done  = 1u;
}


void HAL_SPI_ErrorCallback(SPI_HandleTypeDef *hspi)
{
  if (hspi != &hspi4) {
      return;
  }

  /* Mark DMA no longer active */
  spi_dma_busy = 0u;

  /* Signal error to application */
  spi_dma_error = 1u;
}

/**
 * @brief Public function description
 * @param[in] param1 Description of input parameter
 * @return Description of return value
 */

/* ============================================================================
 * STATIC FUNCTIONS
 * ============================================================================ */

/**
 * @brief Static helper function description
 * @return void
 */
