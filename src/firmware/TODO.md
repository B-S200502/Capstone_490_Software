# Embedded

This folder contains firmware and integration code for the STM32 microcontroller and Raspberry Pi.

---

### Raspberry Pi R&D & Setup
- **EF-01-01: Raspberry Pi Model Selection**  
  Compare Pi 4 vs Pi 5 for performance, power draw, and compatibility.  

- **EF-01-02: Camera Module Research**  
  Evaluate Pi Camera v2 vs v3 for latency, resolution, and ease of integration.  

- **EF-01-03: OS/GPU Acceleration Research**  
  Explore Raspberry Pi OS and GPU acceleration libraries (e.g., OpenCL).  

- **EF-02-01: OS Setup**  
  Install and configure Raspberry Pi OS Bookworm (64-bit headless).  

- **EF-02-02: Driver Installation**  
  Enable camera, GPIO, I²S/SPI/UART interfaces via `dtoverlay` configs.  

- **EF-02-03: Library Installation**  
  Install Python/C++ libraries for audio, video, and DSP (NumPy, OpenCV, ffmpeg).  

---

### STM32 Development
- **EF-03-01: Microphone Array Driver Setup**  
  Configure STM32 ADC/I²S to capture data from all 16 microphones.  

- **EF-03-02: DMA & Buffering**  
  Implement double-buffering with DMA to prevent sample loss.  

- **EF-03-03: Lightweight Preprocessing**  
  Apply offset removal and simple gain normalization before transmission.  

- **EF-04-01: Data Transmission (STM32 to Pi)**  
  Implement efficient packet transfer (I²S/SPI/UART).  
  Include headers or counters for error detection.  

---

### Raspberry Pi Integration
- **EF-04-02: Peripheral Integration**  
  Connect and configure LCD display, battery monitor, and other peripherals.  

- **EF-04-03: System Testing**  
  Measure CPU/GPU utilization during idle, acquisition, and DSP processing.

## STM32 Validation
### ADC Timing + Synchronization
- Are all ADCs triggered when you think they are?
- Is sampling rate correct?
-> We need to design a test where we can calculate the rate that the timer is firing and cross-reference with our settings to ensure that what we are observing matches what we expected
- Are the 4 ADCs aligned frame-to-frame?

#### Tests
- Only enable one ADC?

### DMA correctness
- Half/full callbacks fire at the expected rate
- Buffer layout matches your assumptions (interleaving order!)

### De-interleaving correctness
- Channel ch is really the mic you think it is

### FFT correctness
- Bin frequencies match expected
- Magnitudes / phases behave as expected

### SPI framing correctness
- Packet sizes, header, counter, checksum
- No drops, no misalignment, no endian surprises

### Others
- [ ] Test the USART2 and USB
- [ ] Test that ADC buffers are filling after adc_ready_mask change 
