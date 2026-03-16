#include "audio_preprocessing.hpp"
#include <cstring>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Global preprocessing context
static PreprocessingContext_t preprocessingCtx = {0};
static float processedFrameBuffer[16 * WINDOW_SIZE];

// Window coefficient lookup tables (precomputed for efficiency)
static const float HAMMING_COEFF = 0.54f;
static const float HANNING_COEFF = 0.5f;

PreprocessStatus_t Preprocessing_Init(void)
{
    // Clear context
    memset(&preprocessingCtx, 0, sizeof(preprocessingCtx));
    
    // Initialize default window type
    preprocessingCtx.windowType = WINDOW_HAMMING;
    
    // Generate window coefficients
    PreprocessStatus_t status = Preprocessing_SetWindowType(WINDOW_HAMMING);
    if (status != PREPROCESS_OK) {
        return status;
    }
    
    // Initialize filters for all channels
    for (int ch = 0; ch < 16; ch++) {
        // Design high-pass filter (100 Hz cutoff)
        status = Preprocessing_DesignHighPassFilter(HIGH_PASS_CUTOFF_HZ, 72000.0f, &preprocessingCtx.highPassFilter[ch]);
        if (status != PREPROCESS_OK) {
            return status;
        }
        
        // Design low-pass filter (20 kHz cutoff) 
        status = Preprocessing_DesignLowPassFilter(LOW_PASS_CUTOFF_HZ, 72000.0f, &preprocessingCtx.lowPassFilter[ch]);
        if (status != PREPROCESS_OK) {
            return status;
        }
    }
    
    preprocessingCtx.isInitialized = true;
    preprocessingCtx.frameReady = false;
    
    return PREPROCESS_OK;
}

PreprocessStatus_t Preprocessing_SetWindowType(WindowType_t windowType)
{
    if (!preprocessingCtx.isInitialized && windowType != WINDOW_RECTANGULAR) {
        return PREPROCESS_ERROR;
    }
    
    preprocessingCtx.windowType = windowType;
    
    // Generate window coefficients
    for (int n = 0; n < WINDOW_SIZE; n++) {
        float sample = (float)n;
        float N = (float)WINDOW_SIZE;
        
        switch (windowType) {
            case WINDOW_RECTANGULAR:
                preprocessingCtx.windowCoefficients[n] = 1.0f;
                break;
                
            case WINDOW_HAMMING:
                preprocessingCtx.windowCoefficients[n] = HAMMING_COEFF - (1.0f - HAMMING_COEFF) * cosf(2.0f * M_PI * sample / (N - 1.0f));
                break;
                
            case WINDOW_HANNING:
                preprocessingCtx.windowCoefficients[n] = HANNING_COEFF * (1.0f - cosf(2.0f * M_PI * sample / (N - 1.0f)));
                break;
                
            case WINDOW_BLACKMAN:
                {
                    float a0 = 0.42f;
                    float a1 = 0.5f;
                    float a2 = 0.08f;
                    preprocessingCtx.windowCoefficients[n] = a0 - a1 * cosf(2.0f * M_PI * sample / (N - 1.0f)) + a2 * cosf(4.0f * M_PI * sample / (N - 1.0f));
                }
                break;
                
            default:
                return PREPROCESS_INVALID_PARAMS;
        }
    }
    
    return PREPROCESS_OK;
}

PreprocessStatus_t Preprocessing_ApplyWindow(int16_t* inputData, float* outputData, unsigned int numChannels)
{
    if (!preprocessingCtx.isInitialized || inputData == NULL || outputData == NULL) {
        return PREPROCESS_INVALID_PARAMS;
    }
    
    if (numChannels > 16) {
        return PREPROCESS_INVALID_PARAMS;
    }
    
    // Apply window to each channel
    for (unsigned int ch = 0; ch < numChannels; ch++) {
        for (int n = 0; n < WINDOW_SIZE; n++) {
            int sampleIndex = ch * WINDOW_SIZE + n;
            outputData[sampleIndex] = (float)inputData[sampleIndex] * preprocessingCtx.windowCoefficients[n];
        }
    }
    
    return PREPROCESS_OK;
}

PreprocessStatus_t Preprocessing_ApplyFilters(int16_t* inputData, int16_t* outputData, unsigned int numSamples, unsigned int numChannels)
{
    if (!preprocessingCtx.isInitialized || inputData == NULL || outputData == NULL) {
        return PREPROCESS_INVALID_PARAMS;
    }
    
    if (numChannels > 16 || numSamples != WINDOW_SIZE) {
        return PREPROCESS_INVALID_PARAMS;
    }
    
    // Apply filtering to each channel
    for (unsigned int ch = 0; ch < numChannels; ch++) {
        int16_t* channelInput = &inputData[ch * numSamples];
        int16_t* channelOutput = &outputData[ch * numSamples];
        
        // Apply high-pass filter
        PreprocessStatus_t status = Preprocessing_ApplyIIRFilter(channelInput, channelOutput, numSamples, &preprocessingCtx.highPassFilter[ch]);
        if (status != PREPROCESS_OK) {
            return status;
        }
        
        // Apply low-pass filter (in-place on output)
        status = Preprocessing_ApplyIIRFilter(channelOutput, channelOutput, numSamples, &preprocessingCtx.lowPassFilter[ch]);
        if (status != PREPROCESS_OK) {
            return status;
        }
    }
    
    return PREPROCESS_OK;
}

PreprocessStatus_t Preprocessing_ProcessFrame(int16_t* rawFrame, float* processedFrame)
{
    if (!preprocessingCtx.isInitialized || rawFrame == NULL) {
        return PREPROCESS_INVALID_PARAMS;
    }
    
    // Create temporary buffer for filtered data
    static int16_t filteredFrame[16 * WINDOW_SIZE];
    
    // Step 1: Apply digital filters
    PreprocessStatus_t status = Preprocessing_ApplyFilters(rawFrame, filteredFrame, WINDOW_SIZE, 16);
    if (status != PREPROCESS_OK) {
        return status;
    }
    
    // Step 2: Apply windowing function and convert to float
    if (processedFrame != NULL) {
        status = Preprocessing_ApplyWindow(filteredFrame, processedFrame, 16);
    } else {
        status = Preprocessing_ApplyWindow(filteredFrame, processedFrameBuffer, 16);
    }
    
    if (status == PREPROCESS_OK) {
        preprocessingCtx.frameReady = true;
    }
    
    return status;
}

bool Preprocessing_IsFrameReady(void)
{
    return preprocessingCtx.frameReady;
}

float* Preprocessing_GetProcessedFrame(void)
{
    if (!preprocessingCtx.frameReady) {
        return NULL;
    }
    
    return processedFrameBuffer;
}

void Preprocessing_ReleaseFrame(void)
{
    preprocessingCtx.frameReady = false;
}

PreprocessStatus_t Preprocessing_DesignHighPassFilter(float cutoffHz, float sampleRateHz, IIRFilter_t* filter)
{
    if (filter == NULL || cutoffHz <= 0 || sampleRateHz <= 0) {
        return PREPROCESS_INVALID_PARAMS;
    }
    
    // Simple first-order high-pass filter design
    float omega = 2.0f * M_PI * cutoffHz / sampleRateHz;
    float alpha = 1.0f / (1.0f + tanf(omega / 2.0f));
    
    // High-pass coefficients
    filter->b[0] = alpha;
    filter->b[1] = -alpha;
    filter->a[0] = 1.0f;
    filter->a[1] = alpha - 1.0f;
    
    // Clear delay line
    memset(filter->delay, 0, sizeof(filter->delay));
    
    return PREPROCESS_OK;
}

PreprocessStatus_t Preprocessing_DesignLowPassFilter(float cutoffHz, float sampleRateHz, IIRFilter_t* filter)
{
    if (filter == NULL || cutoffHz <= 0 || sampleRateHz <= 0) {
        return PREPROCESS_INVALID_PARAMS;
    }
    
    // Simple first-order low-pass filter design
    float omega = 2.0f * M_PI * cutoffHz / sampleRateHz;
    float alpha = tanf(omega / 2.0f) / (1.0f + tanf(omega / 2.0f));
    
    // Low-pass coefficients
    filter->b[0] = alpha;
    filter->b[1] = alpha;
    filter->a[0] = 1.0f;
    filter->a[1] = alpha - 1.0f;
    
    // Clear delay line
    memset(filter->delay, 0, sizeof(filter->delay));
    
    return PREPROCESS_OK;
}

PreprocessStatus_t Preprocessing_ApplyIIRFilter(int16_t* input, int16_t* output, unsigned int numSamples, IIRFilter_t* filter)
{
    if (input == NULL || output == NULL || filter == NULL) {
        return PREPROCESS_INVALID_PARAMS;
    }
    
    for (unsigned int n = 0; n < numSamples; n++) {
        float x = (float)input[n];
        float y = filter->b[0] * x + filter->delay[0];
        
        // Update delay line
        filter->delay[0] = filter->b[1] * x - filter->a[1] * y;
        
        // Clamp output to int16_t range
        if (y > 32767.0f) y = 32767.0f;
        if (y < -32768.0f) y = -32768.0f;
        
        output[n] = (int16_t)y;
    }
    
    return PREPROCESS_OK;
}
