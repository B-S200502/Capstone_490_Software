#ifndef FFT_PROCESSOR_H
#define FFT_PROCESSOR_H

#include <stdint.h>
#include <stdbool.h>

// FFT Configuration
#define FFT_SIZE 1024
#define FFT_CHANNELS 16
#define FREQUENCY_BINS (FFT_SIZE / 2)
#define SAMPLE_RATE_HZ 72000

// Complex number structure
typedef struct {
    float real;
    float imag;
} Complex_t;

// FFT output structure for a single channel
typedef struct {
    float magnitude[FREQUENCY_BINS];
    float phase[FREQUENCY_BINS];
    float power_spectrum[FREQUENCY_BINS];
    float frequency_bins[FREQUENCY_BINS];  // Frequency values for each bin
} FFTResult_t;

// Multi-channel FFT result
typedef struct {
    FFTResult_t channels[FFT_CHANNELS];
    bool isValid;
    unsigned int computedChannels;
} MultiChannelFFTResult_t;

// FFT status
typedef enum {
    FFT_OK = 0,
    FFT_ERROR,
    FFT_BUSY,
    FFT_INVALID_SIZE,
    FFT_INVALID_PARAMS,
    FFT_NOT_INITIALIZED
} FFTStatus_t;

// FFT Context
typedef struct {
    Complex_t fftBuffer[FFT_SIZE];
    float twiddle_factors[FFT_SIZE];  // Precomputed twiddle factors for efficiency
    bool isInitialized;
    bool resultReady;
    MultiChannelFFTResult_t currentResult;
} FFTContext_t;

// Function declarations
FFTStatus_t FFT_Init(void);
FFTStatus_t FFT_Compute(float* timeData, FFTResult_t* fftResult, unsigned int channel);
FFTStatus_t FFT_ComputeAllChannels(float* multiChannelData, MultiChannelFFTResult_t* results);
FFTStatus_t FFT_GetPowerSpectrum(FFTResult_t* fftResult, float* powerSpectrum);
FFTStatus_t FFT_GetMagnitudeSpectrum(FFTResult_t* fftResult, float* magnitudeSpectrum);
FFTStatus_t FFT_GetPhaseSpectrum(FFTResult_t* fftResult, float* phaseSpectrum);
bool FFT_IsResultReady(void);
MultiChannelFFTResult_t* FFT_GetResult(void);
void FFT_ReleaseResult(void);

// Utility functions
FFTStatus_t FFT_ComputeFrequencyBins(float sampleRate, float* frequencyBins);
FFTStatus_t FFT_FindPeakFrequency(FFTResult_t* fftResult, float* peakFreq, float* peakMagnitude);
FFTStatus_t FFT_ApplyWindow(float* data, unsigned int length);

// Low-level FFT implementation
FFTStatus_t FFT_Radix2_DIT(Complex_t* data, unsigned int N);
FFTStatus_t FFT_BitReverse(Complex_t* data, unsigned int N);
void FFT_ComputeTwiddleFactors(float* twiddle, unsigned int N);

#endif
