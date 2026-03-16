
# Firmware TODO
**Priority 0**
- [/] Do the check the CLKCNT thing to see how many CCs it takes for FFT
   -> Yep — you’re thinking of the ARM DWT cycle counter, usually accessed through DWT->CYCCNT.


- [ ] (Optional/If beamforming is shit) Calculate and apply per channel phase compensation for mic sampling offset. Can in theory get the exact T(us) between mic samples and apply the phase shift so that each 4 mics per ADC don't have a phase offset from sequential sampling.
**Priority 1**
- [ ] **APP unit testing**
- [ ] Test app_read_battery_millivolts(void) (when we have battery)
- [ ] Investigate bug where sample_rate_hz is getting reset during app_loop()

- [ ] Talk to Rob about the whole audio passthrough (WDYM?)

**Priority 2**
- [ ] Code base clean-up
- [/] DSP unit testing (core done, could add some functions)
- [ ] Verify adc_scaler in dsp_pipeline.c that it should still be 3.3/4096

- [ ] Play with Gain Compensation setting in ADCs in CubeMX to see what it does. Research what we actually require for the beamforming to work
- [ ] Remember: ADC per-channel sampling time is 12.5 CCs, might want to set to 2.5 CCs for release.



- [ ] (QoL) Set up DSP library so it doesn't get removed after CubeMX code generation

- [x] Implement mic IDs
- [x] Find the CubeMX settings to make sure VRef Buf is enabled when code generated -> Run change by Rob to make sure it is conceptually correct
- [x] Evaluate using CMSIS for FFTs and fixed point representation
        -> Added CMSIS DSP to build. Probably not worth to switch to fixed point.
- [x] Add buffer de-interleaving
- [x] Fix adc_ready
- [x] Clean up branches
- [x] Fix the per-channel sampling cycles, make sure all ADCs continuous mode disabled
- [x] Check with a higher timer freq that the ADCs are alive with continuous conversion mode disabled
- [x] Check one ADC with 3 diff resistor combinations (increasing resistance) and check -> CHECKED, MUST FIX GAIN, EITHER 0 OR 3.3V
- [x] Test counter in the main loop for the interrupts
- [x] Bench debug that all the ADCs are firing at the correct rate
- [x] Do some averaging on the data (some stats like peak-to-peak, to see what we get)
- [x] Discuss labeling FFT frames (with counters?)
- [x] Fix USB CDC freeze issue (change usb_printf to non-blocking)
