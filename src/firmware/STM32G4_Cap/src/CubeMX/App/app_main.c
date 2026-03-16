/**
 * @file app_main.c
 * @brief Main application entry point for STM32G4 Acoustic Imager project
 * @details This file contains the main application logic and initialization
 */

/* =========================================================================
 * INCLUDES
 * ========================================================================= */
#include "main.h"
#include "adc.h"
#include "dma.h"
#include "spi.h"
#include "tim.h"
#include "usart.h"
#include "usb_device.h"
#include "gpio.h"

#include "arm_math.h"

#include "app_main.h"

#include "protocol/spi_protocol.h"
#include "transport/spi_stream.h"
#include "transport/spi_stream_test.h"
#include "transport/spi_dma.h"

#include "dsp/dsp_pipeline_test.h"

#include "usb/usb_debug.h"
#include "dsp/dsp_pipeline.h"

#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * DEFINES & CONSTANTS
 * ========================================================================= */
#define ADC_DMA_BUF_SIZE (2 * N_CH_PER_ADC * FRAME_SIZE) // 4096 samples per half-buffer
#define ADC_READY_HALF_MASK ((1u << N_ADCS) - 1u)
#define ADC_READY_FULL_MASK (ADC_READY_HALF_MASK << N_ADCS)
#define PASSTHROUGH_DISABLED_MIC 0xFFu
#define ADC_CLIP_LOW_THRESHOLD 200u
#define ADC_CLIP_HIGH_THRESHOLD 3900u

/* =========================================================================
 * STATIC VARIABLES
 * ========================================================================= */
// Each ADC gets dedicated DMA circular buffer (2 * 4 * 1024 = 8192 samples total)
// Half-buffer = 4096 samples (at 48kHz = 85.2ms)
static uint16_t adc1_buf[ADC_DMA_BUF_SIZE];
static uint16_t adc2_buf[ADC_DMA_BUF_SIZE];
static uint16_t adc3_buf[ADC_DMA_BUF_SIZE];
static uint16_t adc4_buf[ADC_DMA_BUF_SIZE];

static float mic_fft_buffer[FRAME_SIZE];
static float fft_avg[N_MICS][FRAME_SIZE];

// TODO: Test buffer for raw FFT output magnitude calcs, delete or pre-processor guard

static float mag_buffer[FRAME_SIZE/2 + 1];

static arm_rfft_fast_instance_f32 fft_instance;

static spi_stream_t spi_stream_ctx;

static uint32_t _print_counter = 0;
static uint32_t adc_pending_mask = 0;
static uint32_t clip_window_count = 0u;
static uint32_t clip_sample_count = 0u;

// TODO: sample_rate_hz bug
static uint32_t sample_rate_hz = 0;
#define SAMPLE_RATE_HZ 100000

static struct {
  uint8_t enabled;
  uint8_t mic_index;
} passthrough_state = {0u, PASSTHROUGH_DISABLED_MIC};

static char usb_cmd_buffer[APP_USB_CMD_MAX_LEN];
static uint32_t usb_cmd_length = 0u;

// Ping-pong processing flags
// Bit assignment: [4-7]=Full flags (ADC1-4), [0-3]=Half flags (ADC1-4)
volatile uint32_t adc_ready_mask = 0;
volatile uint8_t fft_in_progress = 0;  // Prevent overlapping FFT calculations

// Track which half of buffer is ready for FFT
// ready_half: 0=first half (0-1023), 1=second half (1024-2047)
volatile uint8_t ready_half[4] = {0};  // One per ADC

// Interrupt counters (for debug / watch)
volatile uint32_t irq_events = 0;            // total ADC IRQ events
volatile uint32_t irq_count_adc[4] = {0};    // per-ADC IRQ counters

/* =========================================================================
 * FORWARD DECLARATIONS
 * ========================================================================= */

/* static void module_init(void); */
/* static void module_process(void); */
static uint32_t app_get_tim6_trigger_hz(void);
static void app_process_synced_window(uint32_t half_offset, uint16_t frame_flags);
static const uint16_t *app_get_adc_buffer(uint8_t adc_index);
static uint16_t app_read_battery_millivolts(void);
static uint16_t app_scan_time_domain_clipping(uint32_t half_offset);
static void app_send_passthrough_window(uint32_t half_offset, uint32_t sample_rate_hz);
static void app_handle_usb_command(const char *command);

/* =========================================================================
 * PUBLIC FUNCTIONS
 * ========================================================================= */

// TODO: Deprecated
typedef struct {
    uint8_t header;
    uint8_t adc_id;
    uint32_t frame_counter;
    float fft_data[512]; // TODO: Correct MACRO use
    uint16_t checksum;
} __attribute__((packed)) SPI_Packet_t;

void app_init(void) {
  // Initialize FFT instance (precompute twiddle factors, etc.)
  arm_rfft_fast_init_f32(&fft_instance, FRAME_SIZE);

  // Initialize FFT performance measurement
  init_fft_performance_measurement();
  
  spi_stream_init(&spi_stream_ctx);
}


void app_start(void) {

  // Any additional startup logic can go here
  HAL_Delay(1000);  // Wait for peripherals to stabilize

  // Default LOW gain; then follow Pi's AUTO_GAIN_CNTL (no command yet = pull-down = LOW)
  HAL_GPIO_WritePin(GAIN_CNTL_GPIO_Port, GAIN_CNTL_Pin, GPIO_PIN_RESET);
  GPIO_PinState auto_gain = HAL_GPIO_ReadPin(AUTO_GAIN_CNTL_GPIO_Port, AUTO_GAIN_CNTL_Pin);
  HAL_GPIO_WritePin(GAIN_CNTL_GPIO_Port, GAIN_CNTL_Pin, (auto_gain == GPIO_PIN_SET) ? GPIO_PIN_SET : GPIO_PIN_RESET);

  // Calibrate all ADCs before use
  HAL_ADCEx_Calibration_Start(&hadc1, (uint32_t)ADC_SINGLE_ENDED);
  HAL_ADCEx_Calibration_Start(&hadc2, (uint32_t)ADC_SINGLE_ENDED);
  HAL_ADCEx_Calibration_Start(&hadc3, (uint32_t)ADC_SINGLE_ENDED);
  HAL_ADCEx_Calibration_Start(&hadc4, (uint32_t)ADC_SINGLE_ENDED);


  // Start Timer6 (triggers all ADCs synchronously)
  HAL_TIM_Base_Start(&htim6);
  HAL_Delay(1000);

  sample_rate_hz = app_get_tim6_trigger_hz();
  // Start all 4 ADCs with DMA (2048 samples = 1024×2 ping-pong)
  HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc1_buf, (uint32_t)(ADC_DMA_BUF_SIZE));
  HAL_ADC_Start_DMA(&hadc2, (uint32_t*)adc2_buf, (uint32_t)(ADC_DMA_BUF_SIZE));
  HAL_ADC_Start_DMA(&hadc3, (uint32_t*)adc3_buf, (uint32_t)(ADC_DMA_BUF_SIZE));
  HAL_ADC_Start_DMA(&hadc4, (uint32_t*)adc4_buf, (uint32_t)(ADC_DMA_BUF_SIZE));

  usb_dbg_init(16, 200);  // avg over 16 prints, print every 200 ms

}

/**
 * @brief Main application loop
 * @return void
 */
void app_loop(void) {
  // Poll Pi gain control: read AUTO_GAIN_CNTL and set GAIN_CNTL to match (HIGH/LOW gain)
  GPIO_PinState auto_gain = HAL_GPIO_ReadPin(AUTO_GAIN_CNTL_GPIO_Port, AUTO_GAIN_CNTL_Pin);
  HAL_GPIO_WritePin(GAIN_CNTL_GPIO_Port, GAIN_CNTL_Pin, (auto_gain == GPIO_PIN_SET) ? GPIO_PIN_SET : GPIO_PIN_RESET);

  // Main application loop - check for ADC data ready and process
#if MODE == DEBUG
  if (_print_counter++ >= 1000) {
    _print_counter = 0;
    usb_printf("Main loop heartbeat\r\n");

        
    // Report FFT performance
    usb_printf("FFT Avg Cycles: %d.%02d\r\n", PRINT_F2(get_fft_avg_cycles()));
    usb_printf("FFT Last Cycles: %d.%02d\r\n", PRINT_F2(get_fft_last_cycles()));
  }
#endif

  __disable_irq();
  adc_pending_mask |= adc_ready_mask;
  adc_ready_mask = 0;
  __enable_irq();

  if ((adc_pending_mask & ADC_READY_HALF_MASK) == ADC_READY_HALF_MASK) {
    adc_pending_mask &= ~ADC_READY_HALF_MASK;

    if (!fft_in_progress && !get_spi_dma_busy()) {
      app_process_synced_window(0u, SPI_FRAME_FLAG_SYNCED_ALL_MICS);
    }
    return;
  }

  if ((adc_pending_mask & ADC_READY_FULL_MASK) == ADC_READY_FULL_MASK) {
    adc_pending_mask &= ~ADC_READY_FULL_MASK;

    if (!fft_in_progress && !get_spi_dma_busy()) {
      app_process_synced_window(ADC_DMA_BUF_SIZE / 2u,
                                (uint16_t)(SPI_FRAME_FLAG_SYNCED_ALL_MICS |
                                           SPI_FRAME_FLAG_SECOND_HALF));
    }
    return;
  }
}


  /* Periodic debug print of IRQ counters over UART (every ~1000 iterations) */
  

void test_spi_stream_loop(void) {
  HAL_Delay(1000);
  // spi_stream_unit_test_build_packet();
  // spi_stream_unit_test_nulls();
  // spi_stream_unit_test_small_cap();
  // spi_stream_unit_test_frame_counter();
  // spi_loopback_unit_test();
  spi_loopback_unit_test2();
}

void test_dsp_pipeline_loop(void) {
  HAL_Delay(1000);
  dsp_unit_test_suite();
}

/* ============================================================================
 * STATIC FUNCTIONS
 * ============================================================================ */

/**
 * @brief Static helper function description
 * @return void
 */
static uint32_t app_get_tim6_trigger_hz(void)
{
  uint32_t timer_clock_hz = HAL_RCC_GetPCLK1Freq();

  if ((RCC->CFGR & RCC_CFGR_PPRE1) != RCC_HCLK_DIV1) {
    timer_clock_hz *= 2u;
  }

  return timer_clock_hz / ((htim6.Init.Prescaler + 1u) * (htim6.Init.Period + 1u));
}


static void app_process_synced_window(uint32_t half_offset, uint16_t frame_flags)
{
  // uint32_t sample_rate_hz;
  uint32_t batch_id;
  uint16_t battery_millivolts;
  uint8_t  mic_index;
  size_t   payload_len, final_len;

  uint16_t clipping_hits;

  fft_in_progress = 1u;


  // TODO: Might want to do this once only?
  // sample_rate_hz = app_get_tim6_trigger_hz();
  batch_id = spi_stream_next_batch(&spi_stream_ctx);
  battery_millivolts = app_read_battery_millivolts();

  clipping_hits = app_scan_time_domain_clipping(half_offset);

  if (clipping_hits > 0u) {
    frame_flags |= SPI_FRAME_FLAG_TIME_CLIPPING;
    clip_window_count++;
    clip_sample_count += clipping_hits;
  }

#if MODE != RELEASE
  usb_printf("Clipping hits in window: %u, total clipped windows: %u, total clipped samples: %u\r\n",
             clipping_hits, clip_window_count, clip_sample_count);
#endif

  // Keep mic_index and set to 16 for now
  mic_index = N_MICS;
  payload_len = N_MICS * SPI_MIC_PAYLOAD_BYTES;
  final_len = 0;

  // Need to check here if SPI is done transmitting before starting next batch
  // Use pending flag with spi_dma_done flag?

  uint8_t *tx_buf = spi_dma_get_tx_buffer();
  size_t offset = 0;
  // Fill the header once

  offset += spi_stream_build_frame_header(&spi_stream_ctx,
                                          tx_buf,
                                          SPI_FRAME_PACKET_SIZE_BYTES,
                                          batch_id,
                                          mic_index,
                                          FRAME_SIZE,
                                          SAMPLE_RATE_HZ,
                                          (uint16_t)(frame_flags | SPI_FRAME_FLAG_PAYLOAD_COMPLEX | SPI_FRAME_FLAG_BATTERY_VALID),
                                          payload_len,
                                          battery_millivolts);

  if (offset == 0) {
    // Failed to build header, skip this batch
    fft_in_progress = 0u;
    return;
  }

  // Fill the payload with each mic's FFT data
  for (uint8_t adc = 0; adc < N_ADCS; adc++) {
    uint16_t *active_half = (uint16_t *)(app_get_adc_buffer(adc) + half_offset);

    for (uint8_t channel = 0; channel < N_CH_PER_ADC; channel++) {
      // uint16_t mic_index = (uint16_t)(adc * N_CH_PER_ADC + channel);
      size_t appended_len = 0;
      process_adc_channel_pipeline(&fft_instance,
                                   active_half,
                                   channel,
                                   mic_fft_buffer);

      uint8_t mic_idx = adc * N_CH_PER_ADC + channel;
      update_fft_bin_average(fft_avg[mic_idx], mic_fft_buffer, FRAME_SIZE, FFT_BIN_AVG_BETA);
      
      // if (_print_counter++ == 100) {
      //   _print_counter = 0;

      //   for (int i = 0; i < FRAME_SIZE; i++) {
      //     usb_printf("mic_fft_buffer index=%d, value=%d.%02d\r\n",
      //               i,
      //               PRINT_F3(mic_fft_buffer[i]));
      //     usb_printf("fft_avg        index=%d, value=%d.%02d\r\n",
      //               i,
      //               PRINT_F3(fft_avg[mic_idx][i]));            
      //   }
      //   rfft_packed_to_mag(mic_fft_buffer, mag_buffer, FRAME_SIZE);
      //   dsp_print_fft_report(SAMPLE_RATE_HZ, mag_buffer, FRAME_SIZE);
      //   rfft_packed_to_mag(fft_avg[mic_idx], mag_buffer, FRAME_SIZE);
      //   dsp_print_fft_report(SAMPLE_RATE_HZ, mag_buffer, FRAME_SIZE);
      // }
     
      appended_len = spi_stream_append_mic_payload(tx_buf + offset,
                                                    SPI_FRAME_PACKET_SIZE_BYTES - offset,
                                                    fft_avg[mic_idx],
                                                    FRAME_SIZE);
      
      if (appended_len == 0u) {
        fft_in_progress = 0u;
        return;
      }
      offset += appended_len;                                              
    }
  }

  // Send one big frame, set a pending flag
  final_len = spi_stream_finalize_frame(tx_buf,
                                        SPI_FRAME_PACKET_SIZE_BYTES - offset,
                                        offset);
  
  if (final_len == 0u) {
    fft_in_progress = 0u;
    return;
  }

  if (final_len == SPI_FRAME_PACKET_SIZE_BYTES) {
    // spi_stream_tx_blocking(spi_tx, packet_len);
    spi_stream_tx_dma(tx_buf, SPI_FRAME_PACKET_SIZE_BYTES);
  }

#if MODE != RELEASE
  usb_dbg_push_adc_window_4adc(&adc1_buf[half_offset],
                               &adc2_buf[half_offset],
                               &adc3_buf[half_offset],
                               &adc4_buf[half_offset],
                               FRAME_SIZE);

  if (passthrough_state.enabled) {
    app_send_passthrough_window(half_offset, sample_rate_hz);
  }
#endif

  fft_in_progress = 0u;
}

static const uint16_t *app_get_adc_buffer(uint8_t adc_index)
{
  switch (adc_index) {
    case 0u: return adc1_buf;
    case 1u: return adc2_buf;
    case 2u: return adc3_buf;
    default: return adc4_buf;
  }
}

static uint16_t app_read_battery_millivolts(void)
{
  uint16_t raw_counts = 0u;
  uint32_t sense_mv;

  if (ADC_ReadBatteryRaw(&raw_counts) != HAL_OK) {
    return 0u;
  }

  sense_mv = ((uint32_t)raw_counts * BATT_ADC_VREF_MV) / 4095u;
  sense_mv = (sense_mv * BATT_DIVIDER_NUMERATOR) / BATT_DIVIDER_DENOMINATOR;
  return (uint16_t)sense_mv;
}

static uint16_t app_scan_time_domain_clipping(uint32_t half_offset)
{
  uint32_t hits = 0u;

  for (uint8_t adc = 0u; adc < N_ADCS; adc++) {
    const uint16_t *active_half = app_get_adc_buffer(adc) + half_offset;

    for (uint32_t i = 0u; i < FRAME_SIZE; i++) {
      uint32_t base = i * N_CH_PER_ADC;
      for (uint8_t ch = 0u; ch < N_CH_PER_ADC; ch++) {
        uint16_t sample = active_half[base + ch];
        if (sample <= ADC_CLIP_LOW_THRESHOLD || sample >= ADC_CLIP_HIGH_THRESHOLD) {
          hits++;
        }
      }
    }
  }

  if (hits > 0xFFFFu) {
    return 0xFFFFu;
  }

  return (uint16_t)hits;
}

static void app_send_passthrough_window(uint32_t half_offset, uint32_t sample_rate_hz)
{
  uint8_t mic_index = passthrough_state.mic_index;
  uint8_t adc_index;
  uint8_t adc_channel;
  const uint16_t *adc_window;

  if (mic_index >= N_MICS) {
    return;
  }

  adc_index = mic_index / N_CH_PER_ADC;
  adc_channel = mic_index % N_CH_PER_ADC;
  adc_window = app_get_adc_buffer(adc_index) + half_offset;

  usb_dbg_stream_mic_window_csv(mic_index,
                                adc_window,
                                adc_channel,
                                FRAME_SIZE,
                                sample_rate_hz);
}

void app_usb_cdc_rx(const uint8_t *buf, uint32_t len)
{
  if (!buf) {
    return;
  }

  for (uint32_t i = 0; i < len; i++) {
    char ch = (char)buf[i];

    if (ch == '\r') {
      continue;
    }

    if (ch == '\n') {
      usb_cmd_buffer[usb_cmd_length] = '\0';
      app_handle_usb_command(usb_cmd_buffer);
      usb_cmd_length = 0u;
      continue;
    }

    if (usb_cmd_length + 1u < sizeof(usb_cmd_buffer)) {
      usb_cmd_buffer[usb_cmd_length++] = ch;
    } else {
      usb_cmd_length = 0u;
    }
  }
}

static void app_handle_usb_command(const char *command)
{
  unsigned long mic_index;

  if (!command || command[0] == '\0') {
    return;
  }

  if (strcmp(command, "help") == 0) {
    usb_printf("Commands: help, status, battery, clip, clip reset, pass off, pass <0-15>\r\n");
    return;
  }

  if (strcmp(command, "status") == 0) {
    usb_printf("status: pending=0x%02lX passthrough=%s mic=%u battery=%umV rate=%luHz clip_windows=%lu clip_hits=%lu\r\n",
               (unsigned long)adc_pending_mask,
               passthrough_state.enabled ? "on" : "off",
               (unsigned)passthrough_state.mic_index,
               (unsigned)app_read_battery_millivolts(),
               (unsigned long)app_get_tim6_trigger_hz(),
               (unsigned long)clip_window_count,
               (unsigned long)clip_sample_count);
    return;
  }

  if (strcmp(command, "clip") == 0) {
    usb_printf("clip: windows=%lu hits=%lu thresholds=[<=%u, >=%u]\r\n",
               (unsigned long)clip_window_count,
               (unsigned long)clip_sample_count,
               (unsigned)ADC_CLIP_LOW_THRESHOLD,
               (unsigned)ADC_CLIP_HIGH_THRESHOLD);
    return;
  }

  if (strcmp(command, "clip reset") == 0) {
    clip_window_count = 0u;
    clip_sample_count = 0u;
    usb_printf("clip counters reset\r\n");
    return;
  }

  if (strcmp(command, "battery") == 0) {
    usb_printf("battery=%umV\r\n", (unsigned)app_read_battery_millivolts());
    return;
  }

  if (strcmp(command, "pass off") == 0) {
    passthrough_state.enabled = 0u;
    passthrough_state.mic_index = PASSTHROUGH_DISABLED_MIC;
    usb_printf("passthrough disabled\r\n");
    return;
  }

  if (strncmp(command, "pass ", 5) == 0) {
    mic_index = strtoul(command + 5, NULL, 10);
    if (mic_index < N_MICS) {
      passthrough_state.enabled = 1u;
      passthrough_state.mic_index = (uint8_t)mic_index;
      usb_printf("passthrough mic=%u enabled\r\n", (unsigned)passthrough_state.mic_index);
    } else {
      usb_printf("invalid mic index, expected 0-%u\r\n", (unsigned)(N_MICS - 1u));
    }
    return;
  }

  usb_printf("unknown command: %s\r\n", command);
}

/* Old app_process_synced_window, can be deleted but leaving for now in case */

// static void app_process_synced_window(uint32_t half_offset, uint16_t frame_flags)
// {
//   // uint32_t sample_rate_hz;
//   uint32_t batch_id;
//   uint16_t battery_millivolts;

//   fft_in_progress = 1u;


//   // TODO: Might want to do this once only?
//   // sample_rate_hz = app_get_tim6_trigger_hz();
//   batch_id = spi_stream_next_batch(&spi_stream_ctx);
//   battery_millivolts = app_read_battery_millivolts();

//   for (uint8_t adc = 0; adc < N_ADCS; adc++) {
//     uint16_t *active_half = (uint16_t *)(app_get_adc_buffer(adc) + half_offset);

//     for (uint8_t channel = 0; channel < N_CH_PER_ADC; channel++) {
//       uint16_t mic_index = (uint16_t)(adc * N_CH_PER_ADC + channel);
//       size_t packet_len;
      
//       process_adc_channel_pipeline(&fft_instance,
//                                    active_half,
//                                    channel,
//                                    mic_fft_buffer);
      
//       // if (_print_counter++ == 100) {
//       // _print_counter = 0;

//       // for (int i = 0; i < FRAME_SIZE; i++) {
//       //   usb_printf("index=%d, value=%d.%02d\r\n",
//       //             i,
//       //             PRINT_F3(mic_fft_buffer[i]));          
//       // }
//       // rfft_packed_to_mag(mic_fft_buffer, mag_buffer, FRAME_SIZE);
//       // dsp_print_fft_report(SAMPLE_RATE_HZ, mag_buffer, FRAME_SIZE);
//       // }
     
      
//       uint8_t *tx_buf = spi_dma_get_tx_buffer();

//       packet_len = spi_stream_build_mic_packet(
//           &spi_stream_ctx,
//           tx_buf,
//           SPI_PACKET_SIZE,
//           batch_id,
//           mic_index,
//           mic_fft_buffer,
//           FRAME_SIZE,
//           sample_rate_hz,
//           (uint16_t)(frame_flags | SPI_FRAME_FLAG_PAYLOAD_COMPLEX | SPI_FRAME_FLAG_BATTERY_VALID),
//           battery_millivolts);

//       if (packet_len > 0u) {
//         // spi_stream_tx_blocking(spi_tx, packet_len);
//         spi_stream_tx_dma(tx_buf, packet_len);
//       }
//     }
//   }

//   usb_dbg_push_adc_window_4adc(&adc1_buf[half_offset],
//                                &adc2_buf[half_offset],
//                                &adc3_buf[half_offset],
//                                &adc4_buf[half_offset],
//                                FRAME_SIZE);

//   if (passthrough_state.enabled) {
//     app_send_passthrough_window(half_offset, sample_rate_hz);
//   }

//   fft_in_progress = 0u;
// }
  