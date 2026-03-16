#!/usr/bin/env python3
"""
Acoustic Imager - Heatmap Pipeline Test Script

This script simulates the heatmap pipeline:
1. Generates fake heatmap data (NumPy arrays)
2. Converts arrays to colored heatmaps using OpenCV
3. Overlays heatmaps onto background frames with alpha blending
4. Displays results in real-time and streams to FFmpeg for recording

"""

import numpy as np
import cv2
import subprocess
import sys
import time
from typing import Tuple, Optional


def generate_fake_heatmap(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Generate a fake heatmap array to simulate audio data.
    
    Args:
        width: Width of the heatmap in pixels
        height: Height of the heatmap in pixels
        
    Returns:
        NumPy array with random values between 0-255 representing heatmap intensity
    """
    # Create a random 2D array with some spatial correlation to simulate realistic heatmap
    # Use a combination of random noise and smooth patterns
    base_noise = np.random.normal(128, 50, (height, width))
    
    # Add some smooth patterns to make it more realistic
    x = np.linspace(0, 4*np.pi, width)
    y = np.linspace(0, 4*np.pi, height)
    X, Y = np.meshgrid(x, y)
    pattern = 30 * np.sin(X) * np.cos(Y)
    
    # Combine noise and pattern
    heatmap_data = base_noise + pattern
    
    # Clamp values to valid range [0, 255] and convert to uint8
    heatmap_data = np.clip(heatmap_data, 0, 255).astype(np.uint8)
    
    return heatmap_data


def apply_heatmap_overlay(heatmap_data: np.ndarray, 
                         background_frame: np.ndarray, 
                         alpha: float = 0.7) -> np.ndarray:
    """
    Convert heatmap data to colored heatmap and overlay it on background frame.
    
    Args:
        heatmap_data: 2D NumPy array with heatmap intensity values
        background_frame: Background frame to overlay heatmap onto
        alpha: Transparency factor for blending (0.0 = transparent, 1.0 = opaque)
        
    Returns:
        Combined frame with heatmap overlay
    """
    # Apply color map to convert grayscale heatmap to colored heatmap
    # Using JET colormap for classic heatmap appearance
    colored_heatmap = cv2.applyColorMap(heatmap_data, cv2.COLORMAP_JET)
    
    # Ensure both images have the same dimensions
    # TODO: Will be important when we get the camera feed, overlay needs to be the same size 
    if colored_heatmap.shape[:2] != background_frame.shape[:2]:
        colored_heatmap = cv2.resize(colored_heatmap, 
                                   (background_frame.shape[1], background_frame.shape[0]))
    
    # Perform alpha blending (technique for combining images with transparency)
    # Formula: result = alpha * heatmap + (1 - alpha) * background
    blended_frame = cv2.addWeighted(colored_heatmap, alpha, background_frame, 1 - alpha, 0)
    
    return blended_frame


def create_background_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create a background frame for overlay.
    
    Args:
        width: Width of the frame
        height: Height of the frame
        
    Returns:
        Background frame (black by default)
    """
    # Create a black background frame
    background = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Optionally add some visual elements to make it more interesting
    # Add a subtle grid pattern
    for i in range(0, height, 40):
        cv2.line(background, (0, i), (width, i), (20, 20, 20), 1)
    for i in range(0, width, 40):
        cv2.line(background, (i, 0), (i, height), (20, 20, 20), 1)
    
    return background


def send_to_ffmpeg(frame: np.ndarray, ffmpeg_process: subprocess.Popen) -> None:
    """
    Send a frame to FFmpeg subprocess for recording.
    
    Args:
        frame: Frame to send (BGR format)
        ffmpeg_process: Active FFmpeg subprocess
    """
    try:
        # Convert frame to bytes and write to FFmpeg stdin
        frame_bytes = frame.tobytes()
        ffmpeg_process.stdin.write(frame_bytes)
        ffmpeg_process.stdin.flush()
    except (BrokenPipeError, OSError) as e:
        print(f"Error writing to FFmpeg: {e}")
        print("FFmpeg process may have terminated unexpectedly.")


def setup_ffmpeg_process(width: int = 640, height: int = 480, 
                        fps: int = 30, output_file: str = "output.mp4") -> subprocess.Popen:
    """
    Set up FFmpeg subprocess for hardware-accelerated recording.
    
    Args:
        width: Video width
        height: Video height
        fps: Frames per second
        output_file: Output video file path
        
    Returns:
        FFmpeg subprocess with stdin pipe
    """
    import platform
    
    # Choose encoder based on platform
    system = platform.system().lower()
    if system == "darwin":  # macOS
        # Use VideoToolbox hardware encoder on macOS
        encoder = 'h264_videotoolbox'
    elif system == "linux":  # Linux/Raspberry Pi
        # Use v4l2m2m hardware encoder on Linux
        encoder = 'h264_v4l2m2m'
    else:  # Windows or other
        # Use software encoder as fallback
        encoder = 'libx264'
    
    # FFmpeg command for hardware-accelerated recording
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file
        '-f', 'rawvideo',  # Input format
        '-pix_fmt', 'bgr24',  # Pixel format (BGR for OpenCV)
        '-s', f'{width}x{height}',  # Video size
        '-r', str(fps),  # Frame rate
        '-i', '-',  # Read from stdin
        '-c:v', encoder,  # Platform-specific encoder
        '-b:v', '2M',  # Bitrate
        output_file
    ]
    
    try:
        # Start FFmpeg subprocess
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"FFmpeg process started with {encoder} encoder. Recording to: {output_file}")
        return process
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting FFmpeg process: {e}")
        sys.exit(1)


def cleanup_ffmpeg_process(ffmpeg_process: subprocess.Popen) -> None:
    """
    Clean up FFmpeg subprocess.
    
    Args:
        ffmpeg_process: FFmpeg subprocess to clean up
    """
    try:
        # Close stdin to signal end of input
        ffmpeg_process.stdin.close()
        
        # Wait for process to complete
        stdout, stderr = ffmpeg_process.communicate(timeout=10)
        
        if ffmpeg_process.returncode == 0:
            print("FFmpeg recording completed successfully.")
        else:
            print(f"FFmpeg process exited with code {ffmpeg_process.returncode}")
            if stderr:
                print(f"FFmpeg stderr: {stderr.decode()}")
                
    except subprocess.TimeoutExpired:
        print("FFmpeg process timed out. Force killing...")
        ffmpeg_process.kill()
    except Exception as e:
        print(f"Error during FFmpeg cleanup: {e}")


def main():
    print("Acoustic Imager - Heatmap Pipeline Test")
    print("=" * 50)
    
    # Configuration
    WIDTH = 640
    HEIGHT = 480
    FPS = 30
    NUM_FRAMES = 100
    ALPHA = 0.7  # Heatmap transparency
    OUTPUT_FILE = "heatmap_output.mp4"
    
    # Create background frame
    print("Creating background frame...")
    background = create_background_frame(WIDTH, HEIGHT)
    
    # Set up FFmpeg process
    print("Setting up FFmpeg process...")
    ffmpeg_process = setup_ffmpeg_process(WIDTH, HEIGHT, FPS, OUTPUT_FILE)
    
    # Create OpenCV window for real-time display
    cv2.namedWindow('Acoustic Imager - Heatmap Pipeline', cv2.WINDOW_AUTOSIZE)
    
    try:
        print(f"Starting simulation with {NUM_FRAMES} frames...")
        print("Press 'q' to quit early, or wait for completion.")
        
        start_time = time.time()
        
        for frame_num in range(NUM_FRAMES):
            # Generate fake heatmap data
            heatmap_data = generate_fake_heatmap(WIDTH, HEIGHT)
            
            # Apply heatmap overlay
            output_frame = apply_heatmap_overlay(heatmap_data, background, ALPHA)
            
            # Add frame number and timestamp to the frame
            timestamp = time.time() - start_time
            cv2.putText(output_frame, f"Frame: {frame_num + 1}/{NUM_FRAMES}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(output_frame, f"Time: {timestamp:.2f}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame in real-time
            cv2.imshow('Acoustic Imager - Heatmap Pipeline', output_frame)
            
            # Send frame to FFmpeg for recording
            send_to_ffmpeg(output_frame, ffmpeg_process)
            
            # Control frame rate (approximate)
            frame_delay = 1000 // FPS  # Convert to milliseconds
            key = cv2.waitKey(frame_delay) & 0xFF
            
            # Check for quit command
            if key == ord('q'):
                print("Quit requested by user.")
                break
                
            # Print progress every 10 frames
            if (frame_num + 1) % 10 == 0:
                print(f"Processed {frame_num + 1}/{NUM_FRAMES} frames...")
        
        elapsed_time = time.time() - start_time
        print(f"Simulation completed in {elapsed_time:.2f} seconds.")
        print(f"Average FPS: {NUM_FRAMES / elapsed_time:.2f}")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        # Cleanup
        print("Cleaning up...")
        cv2.destroyAllWindows()
        cleanup_ffmpeg_process(ffmpeg_process)
        print(f"Output video saved as: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
