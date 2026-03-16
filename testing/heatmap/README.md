# Acoustic Imager - Heatmap Pipeline Test

This directory contains the test script for the Acoustic Imager heatmap pipeline, which simulates real-time audio visualization using heatmaps overlaid on video frames.

## Overview

The heatmap pipeline test script demonstrates the core functionality of the Acoustic Imager system:

1. **Fake Data Generation**: Creates realistic heatmap data using NumPy arrays
2. **Heatmap Visualization**: Converts data to colored heatmaps using OpenCV
3. **Frame Overlay**: Blends heatmaps onto background frames with alpha transparency
4. **Real-time Display**: Shows results in an OpenCV window
5. **FFmpeg Recording**: Streams frames to FFmpeg for hardware-accelerated video recording

## Files

- `heatmap_pipeline_test.py` - Main test script implementing the complete pipeline
- `requirements.txt` - Python dependencies
- `README.md` - This documentation file

## Requirements

### Python Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### System Dependencies
- **FFmpeg**: Must be installed on your system for video recording
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use `winget install ffmpeg`

## Usage

Run the test script:

**On macOS/Linux:**
```bash
python3 heatmap_pipeline_test.py
```

**On Windows:**
```cmd
python heatmap_pipeline_test.py
```

### Controls
- **q**: Quit the simulation early
- **Close window**: Exit the program

### Output
- Real-time display window showing the heatmap overlay
- Video file: `heatmap_output.mp4` (saved in the same directory)

## Configuration

You can modify these parameters in the `main()` function:

- `WIDTH`, `HEIGHT`: Video dimensions (default: 640x480)
- `FPS`: Frames per second (default: 30)
- `NUM_FRAMES`: Number of frames to generate (default: 100)
- `ALPHA`: Heatmap transparency (0.0-1.0, default: 0.7)
- `OUTPUT_FILE`: Output video filename

## Architecture

The script is organized into modular functions:

### Core Functions
- `generate_fake_heatmap()`: Creates random heatmap data with spatial correlation
- `apply_heatmap_overlay()`: Converts data to colored heatmap and blends with background
- `send_to_ffmpeg()`: Streams frames to FFmpeg subprocess
- `setup_ffmpeg_process()`: Initializes FFmpeg with hardware acceleration
- `cleanup_ffmpeg_process()`: Properly closes FFmpeg subprocess

### FFmpeg Configuration
The script automatically selects the appropriate encoder based on your platform:

**macOS**: Uses `h264_videotoolbox` (hardware-accelerated)
**Linux/Raspberry Pi**: Uses `h264_v4l2m2m` (hardware-accelerated)  
**Windows**: Uses `libx264` (software encoder)

Example FFmpeg command structure:
```bash
ffmpeg -y -f rawvideo -pix_fmt bgr24 -s 640x480 -r 30 -i - \
  -c:v h264_v4l2m2m -b:v 2M output.mp4
```

This configuration:
- Uses `h264_v4l2m2m` encoder for Raspberry Pi hardware acceleration
- Accepts raw BGR24 video input from stdin
- Records at 2Mbps bitrate
- Outputs to MP4 format

## Integration Notes

This test script simulates the heatmap pipeline that will be integrated into the full Acoustic Imager system. In the actual implementation:

1. **Audio Processing**: Replace `generate_fake_heatmap()` with real audio analysis
2. **Real-time Performance**: Optimize for actual audio processing latency
3. **Hardware Integration**: Adapt FFmpeg settings for specific hardware platforms
4. **Error Handling**: Add robust error handling for production use

## Troubleshooting

### Common Issues
1. **FFmpeg not found**: Ensure FFmpeg is installed and in your system PATH
2. **Hardware encoder not available**: The `h264_v4l2m2m` encoder is specific to Raspberry Pi. On other systems, use `libx264` instead
3. **Permission errors**: Ensure write permissions for the output directory

### Performance Notes
- The script includes frame rate control to maintain consistent timing
- FFmpeg subprocess is properly managed with cleanup on exit
- Memory usage is optimized by processing frames one at a time
