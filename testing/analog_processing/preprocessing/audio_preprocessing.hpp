#ifndef AUDIO_PREPROCESSING_H
#define AUDIO_PREPROCESSING_H

#include <stdint.h>
#include <stdbool.h>

// Preprocessing configuration
#define WINDOW_SIZE 1024
#define OVERLAP_FACTOR 0.5f
#define HIGH_PASS_CUTOFF_HZ 100
#define LOW_PASS_CUTOFF_HZ 20000
#define FILTER_ORDER 4

// Window types for FFT preprocessing
typedef enum {
    WINDOW_RECTANGULAR = 0,
    WINDOW_HAMMING,
    WINDOW_HANNING,
    WINDOW_BLACKMAN
} WindowType_t;

// Filter types
typedef enum {
    FILTER_HIGH_PASS,
    FILTER_LOW_PASS,
    FILTER_BAND_PASS,
    FILTER_NOTCH
} FilterType_t;

// Preprocessing status
typedef enum {
    PREPROCESS_OK = 0,
    PREPROCESS_ERROR,
    PREPROCESS_INVALID_PARAMS,
    PREPROCESS_BUSY
} PreprocessStatus_t;

// Filter coefficients structure
typedef struct {
    float b[FILTER_ORDER + 1];  // Numerator coefficients
    float a[FILTER_ORDER + 1];  // Denominator coefficients
    float delay[FILTER_ORDER];  // Delay line
} IIRFilter_t;

// Preprocessing context
typedef struct {
    IIRFilter_t highPassFilter[16];  // One filter per channel
    IIRFilter_t lowPassFilter[16];   // One filter per channel
    float windowCoefficients[WINDOW_SIZE];
    WindowType_t windowType;
    bool isInitialized;
    bool frameReady;
    int16_t processedFrame[16 * WINDOW_SIZE];  // 16 channels * window size
} PreprocessingContext_t;

// Function declarations
PreprocessStatus_t Preprocessing_Init(void);
PreprocessStatus_t Preprocessing_SetWindowType(WindowType_t windowType);
PreprocessStatus_t Preprocessing_ApplyWindow(int16_t* inputData, float* outputData, unsigned int numChannels);
PreprocessStatus_t Preprocessing_ApplyFilters(int16_t* inputData, int16_t* outputData, unsigned int numSamples, unsigned int numChannels);
PreprocessStatus_t Preprocessing_Normalize(int16_t* inputData, int16_t* outputData, unsigned int numSamples, unsigned int numChannels);
PreprocessStatus_t Preprocessing_ProcessFrame(int16_t* rawFrame, float* processedFrame);
bool Preprocessing_IsFrameReady(void);
float* Preprocessing_GetProcessedFrame(void);
void Preprocessing_ReleaseFrame(void);

// Utility functions
PreprocessStatus_t Preprocessing_DesignHighPassFilter(float cutoffHz, float sampleRateHz, IIRFilter_t* filter);
PreprocessStatus_t Preprocessing_DesignLowPassFilter(float cutoffHz, float sampleRateHz, IIRFilter_t* filter);
PreprocessStatus_t Preprocessing_ApplyIIRFilter(int16_t* input, int16_t* output, unsigned int numSamples, IIRFilter_t* filter);

#endif
