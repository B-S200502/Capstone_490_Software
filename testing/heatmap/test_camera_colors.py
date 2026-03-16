#!/usr/bin/env python3
"""
Camera Color Test Script
Tests different color formats and white balance modes to fix blue hue issue
"""

import cv2
import numpy as np
import time

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    print("ERROR: picamera2 not available")
    PICAMERA2_AVAILABLE = False
    exit(1)

WIDTH = 640
HEIGHT = 480

# White balance modes to test
AWB_MODES = {
    0: "Auto",
    1: "Tungsten (Indoor/Warm)",
    2: "Fluorescent",
    3: "Indoor",
    4: "Daylight",
    5: "Cloudy",
}

def test_camera_with_awb_mode(awb_mode, mode_name):
    """Test camera with specific white balance mode"""
    print(f"\n{'='*60}")
    print(f"Testing AWB Mode {awb_mode}: {mode_name}")
    print(f"{'='*60}")
    
    picam2 = Picamera2()
    
    # Configure camera
    config = picam2.create_preview_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    
    # Set white balance mode
    try:
        picam2.set_controls({
            "AwbEnable": True,
            "AwbMode": awb_mode
        })
        print(f"✓ White balance mode set to: {mode_name}")
    except Exception as e:
        print(f"✗ Failed to set white balance: {e}")
        picam2.close()
        return
    
    picam2.start()
    
    # Wait for camera to adjust
    print("Waiting for camera to adjust (3 seconds)...")
    time.sleep(3)
    
    # Create window
    window_name = f"AWB Mode {awb_mode}: {mode_name} (Press SPACE to save, Q to next)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    print("Press SPACE to save a test image, Q to move to next mode")
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Convert RGB to BGR for OpenCV display
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add text overlay
            cv2.putText(frame_bgr, f"AWB Mode: {awb_mode} - {mode_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_bgr, "SPACE=Save  Q=Next Mode", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display
            cv2.imshow(window_name, frame_bgr)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("Moving to next mode...")
                break
            elif key == ord(' '):
                # Save test image
                filename = f"camera_test_awb_{awb_mode}_{mode_name.replace(' ', '_')}.png"
                cv2.imwrite(filename, frame_bgr)
                print(f"✓ Saved: {filename}")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cv2.destroyWindow(window_name)
        picam2.stop()
        picam2.close()
        print(f"Closed AWB mode {awb_mode}")

def main():
    print("Camera Color Test - Blue Hue Diagnosis")
    print("="*60)
    print("This script will test different white balance modes")
    print("to help you find the best setting to fix the blue hue.")
    print("="*60)
    
    if not PICAMERA2_AVAILABLE:
        print("ERROR: picamera2 not available")
        return
    
    # Test each white balance mode
    for awb_mode, mode_name in AWB_MODES.items():
        test_camera_with_awb_mode(awb_mode, mode_name)
        time.sleep(0.5)  # Brief pause between modes
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)
    print("\nReview the saved images to find which AWB mode looks best.")
    print("Then update heatmap_new_features.py line ~948:")
    print('  "AwbMode": X  # Replace X with the best mode number')
    print("\nMode recommendations:")
    print("  - Blue tint → Try mode 1 (Tungsten) or 3 (Indoor)")
    print("  - Yellow tint → Try mode 4 (Daylight) or 5 (Cloudy)")
    print("  - Inconsistent → Try mode 0 (Auto)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
