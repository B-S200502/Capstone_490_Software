
# Firmware TODO

- [ ] APP unit testing
- [/] DSP unit testing (core done, could add some functions)

- [ ] Do the check the CLKCNT thing to see how many CCs it takes for FFT
- [ ] Investigate bug where sample_rate_hz is getting reset during app_loop()

- [ ] Verify adc_scaler in dsp_pipeline.c that it is still 3.3/4096

- [ ] Play with Gain Compensation setting in ADCs in CubeMX to see what it does. Research what we actually require for the beamforming to work
- [ ] Address that ADCs are configured properly for channel scanning in CubeMX

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
