#include "fft_processor.hpp"
#include <cstring>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Global FFT context
static FFTContext_t fftCtx = {0};

FFTStatus_t FFT_Init(void)
{
    // Clear context
    memset(&fftCtx, 0, sizeof(fftCtx));
    
    // Precompute twiddle factors for efficiency
    FFT_ComputeTwiddleFactors(fftCtx.twiddle_factors, FFT_SIZE);
    
    // Precompute frequency bins
    for (int i = 0; i < FFT_CHANNELS; i++) {
        FFT_ComputeFrequencyBins(SAMPLE_RATE_HZ, fftCtx.currentResult.channels[i].frequency_bins);
    }
    
    fftCtx.isInitialized = true;
    fftCtx.resultReady = false;
    fftCtx.currentResult.isValid = false;
    fftCtx.currentResult.computedChannels = 0;
    
    return FFT_OK;
}

FFTStatus_t FFT_Compute(float* timeData, FFTResult_t* fftResult, unsigned int channel)
{
    if (!fftCtx.isInitialized || timeData == NULL || fftResult == NULL) {
        return FFT_INVALID_PARAMS;
    }
    
    if (channel >= FFT_CHANNELS) {
        return FFT_INVALID_PARAMS;
    }
    
    // Copy time domain data to complex buffer (real part only)
    for (int i = 0; i < FFT_SIZE; i++) {
        fftCtx.fftBuffer[i].real = timeData[i];
        fftCtx.fftBuffer[i].imag = 0.0f;
    }
    
    // Perform FFT
    FFTStatus_t status = FFT_Radix2_DIT(fftCtx.fftBuffer, FFT_SIZE);
    if (status != FFT_OK) {
        return status;
    }
    
    // Compute magnitude, phase, and power spectrum (only for positive frequencies)
    for (int i = 0; i < FREQUENCY_BINS; i++) {
        float real = fftCtx.fftBuffer[i].real;
        float imag = fftCtx.fftBuffer[i].imag;
        
        // Magnitude spectrum
        fftResult->magnitude[i] = sqrtf(real * real + imag * imag);
        
        // Phase spectrum
        fftResult->phase[i] = atan2f(imag, real);
        
        // Power spectrum (magnitude squared)
        fftResult->power_spectrum[i] = fftResult->magnitude[i] * fftResult->magnitude[i];
        
        // Normalize by FFT size
        if (i == 0 || i == FREQUENCY_BINS - 1) {
            // DC and Nyquist components
            fftResult->magnitude[i] /= FFT_SIZE;
            fftResult->power_spectrum[i] /= (FFT_SIZE * FFT_SIZE);
        } else {
            // All other components (multiply by 2 for single-sided spectrum)
            fftResult->magnitude[i] *= 2.0f / FFT_SIZE;
            fftResult->power_spectrum[i] *= 4.0f / (FFT_SIZE * FFT_SIZE);
        }
    }
    
    // Copy frequency bins
    memcpy(fftResult->frequency_bins, fftCtx.currentResult.channels[0].frequency_bins, sizeof(fftResult->frequency_bins));
    
    return FFT_OK;
}

FFTStatus_t FFT_ComputeAllChannels(float* multiChannelData, MultiChannelFFTResult_t* results)
{
    if (!fftCtx.isInitialized || multiChannelData == NULL || results == NULL) {
        return FFT_INVALID_PARAMS;
    }
    
    results->isValid = false;
    results->computedChannels = 0;
    
    // Process each channel
    for (int ch = 0; ch < FFT_CHANNELS; ch++) {
        float* channelData = &multiChannelData[ch * FFT_SIZE];
        
        FFTStatus_t status = FFT_Compute(channelData, &results->channels[ch], ch);
        if (status != FFT_OK) {
            return status;
        }
        
        results->computedChannels++;
    }
    
    results->isValid = true;
    
    // Store result in context
    memcpy(&fftCtx.currentResult, results, sizeof(MultiChannelFFTResult_t));
    fftCtx.resultReady = true;
    
    return FFT_OK;
}

FFTStatus_t FFT_GetPowerSpectrum(FFTResult_t* fftResult, float* powerSpectrum)
{
    if (fftResult == NULL || powerSpectrum == NULL) {
        return FFT_INVALID_PARAMS;
    }
    
    memcpy(powerSpectrum, fftResult->power_spectrum, FREQUENCY_BINS * sizeof(float));
    return FFT_OK;
}

FFTStatus_t FFT_GetMagnitudeSpectrum(FFTResult_t* fftResult, float* magnitudeSpectrum)
{
    if (fftResult == NULL || magnitudeSpectrum == NULL) {
        return FFT_INVALID_PARAMS;
    }
    
    memcpy(magnitudeSpectrum, fftResult->magnitude, FREQUENCY_BINS * sizeof(float));
    return FFT_OK;
}

FFTStatus_t FFT_GetPhaseSpectrum(FFTResult_t* fftResult, float* phaseSpectrum)
{
    if (fftResult == NULL || phaseSpectrum == NULL) {
        return FFT_INVALID_PARAMS;
    }
    
    memcpy(phaseSpectrum, fftResult->phase, FREQUENCY_BINS * sizeof(float));
    return FFT_OK;
}

bool FFT_IsResultReady(void)
{
    return fftCtx.resultReady;
}

MultiChannelFFTResult_t* FFT_GetResult(void)
{
    if (!fftCtx.resultReady) {
        return NULL;
    }
    
    return &fftCtx.currentResult;
}

void FFT_ReleaseResult(void)
{
    fftCtx.resultReady = false;
}

FFTStatus_t FFT_ComputeFrequencyBins(float sampleRate, float* frequencyBins)
{
    if (frequencyBins == NULL || sampleRate <= 0) {
        return FFT_INVALID_PARAMS;
    }
    
    float df = sampleRate / FFT_SIZE;  // Frequency resolution
    
    for (int i = 0; i < FREQUENCY_BINS; i++) {
        frequencyBins[i] = i * df;
    }
    
    return FFT_OK;
}

FFTStatus_t FFT_FindPeakFrequency(FFTResult_t* fftResult, float* peakFreq, float* peakMagnitude)
{
    if (fftResult == NULL || peakFreq == NULL || peakMagnitude == NULL) {
        return FFT_INVALID_PARAMS;
    }
    
    int peakIndex = 0;
    float maxMagnitude = fftResult->magnitude[0];
    
    // Find peak (skip DC component at index 0)
    for (int i = 1; i < FREQUENCY_BINS; i++) {
        if (fftResult->magnitude[i] > maxMagnitude) {
            maxMagnitude = fftResult->magnitude[i];
            peakIndex = i;
        }
    }
    
    *peakFreq = fftResult->frequency_bins[peakIndex];
    *peakMagnitude = maxMagnitude;
    
    return FFT_OK;
}

// Low-level FFT implementation using Radix-2 Decimation-in-Time
FFTStatus_t FFT_Radix2_DIT(Complex_t* data, unsigned int N)
{
    if (data == NULL || N == 0 || (N & (N - 1)) != 0) {
        return FFT_INVALID_SIZE;  // N must be power of 2
    }
    
    // Bit-reverse the input
    FFTStatus_t status = FFT_BitReverse(data, N);
    if (status != FFT_OK) {
        return status;
    }
    
    // Perform FFT stages
    for (unsigned int stage = 1; stage <= (unsigned int)log2f(N); stage++) {
        unsigned int m = 1 << stage;  // 2^stage
        unsigned int m2 = m >> 1;     // m/2
        
        // Twiddle factor
        Complex_t w = {1.0f, 0.0f};
        Complex_t wm = {cosf(-2.0f * M_PI / m), sinf(-2.0f * M_PI / m)};
        
        for (unsigned int j = 0; j < m2; j++) {
            for (unsigned int k = j; k < N; k += m) {
                unsigned int k2 = k + m2;
                
                // Butterfly computation
                Complex_t t = {
                    w.real * data[k2].real - w.imag * data[k2].imag,
                    w.real * data[k2].imag + w.imag * data[k2].real
                };
                
                data[k2].real = data[k].real - t.real;
                data[k2].imag = data[k].imag - t.imag;
                data[k].real = data[k].real + t.real;
                data[k].imag = data[k].imag + t.imag;
            }
            
            // Update twiddle factor
            Complex_t temp = w;
            w.real = temp.real * wm.real - temp.imag * wm.imag;
            w.imag = temp.real * wm.imag + temp.imag * wm.real;
        }
    }
    
    return FFT_OK;
}

FFTStatus_t FFT_BitReverse(Complex_t* data, unsigned int N)
{
    if (data == NULL || N == 0) {
        return FFT_INVALID_PARAMS;
    }
    
    unsigned int j = 0;
    for (unsigned int i = 1; i < N; i++) {
        unsigned int bit = N >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        
        if (i < j) {
            // Swap data[i] and data[j]
            Complex_t temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
    
    return FFT_OK;
}

void FFT_ComputeTwiddleFactors(float* twiddle, unsigned int N)
{
    for (unsigned int i = 0; i < N; i++) {
        twiddle[i] = cosf(-2.0f * M_PI * i / N);
    }
}
