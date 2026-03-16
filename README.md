# Acoustic Imager - Real-Time Sound Localization System

## Overview

This repository contains a complete **acoustic imaging system** developed as a student capstone project (COEN 490). The system uses a **16-microphone array** to localize sound sources and generate visual heatmaps in real-time, combining advanced signal processing with interactive visualization.

**Key Features:**
- **Custom Fermat spiral array geometry** for optimal angular resolution
- **Real-time beamforming** using MUSIC and ESPRIT algorithms
- **Interactive visualization** with camera overlay and bandpass filtering
- **Dual data sources**: Simulated signals and SPI hardware interface
- **Professional UI** with video recording, screenshots, and performance monitoring

---

## Project Objectives

- Achieve **~9° angular resolution** with a 16-mic array
- Manage spatial aliasing above ~8.6 kHz using **subspace DSP algorithms**
- Implement **low-level acquisition on STM32** with data forwarding to Raspberry Pi
- Provide **real-time visualization** of sound fields with interactive controls
- Deliver a **working prototype** with modular, maintainable software architecture

---

## Team

- **David** – DSP lead, MUSIC/ESPRIT algorithms, visualization
- **Basem** – C++ DSP implementation, software architecture
- **Fabian** – Requirements, integration testing
- **Tim** – Literature review, peripheral testing
- **Rob** – Hardware selection, PCB design
- **Ahmad** – Housing, assembly, CAD

---

## Hardware Components

### Microphone Array
- **Configuration**: 16 MEMS microphones in Fermat spiral layout
- **Aperture**: ~8 cm diameter
- **Geometry**: Golden angle spacing (~137.5°) for isotropic coverage
- **Frequency Range**: 0-36 kHz (with aliasing management above 8.6 kHz)

### Processing Hardware
- **STM32 Microcontroller**: Low-level data acquisition and preprocessing
- **Raspberry Pi 4**: Real-time DSP, beamforming, and visualization
- **SPI Interface**: High-speed data transfer between STM32 and Pi

### Optional Components
- **Camera**: Raspberry Pi Camera Module or USB camera for background overlay
- **Display**: HDMI monitor for real-time visualization

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/Capstone_490_Software.git
cd Capstone_490_Software

# Install Python dependencies
pip3 install numpy opencv-python

# Optional: Install hardware support
pip3 install spidev picamera2
```

### Running the Application

**Method 1: From software directory (recommended)**
```bash
cd src/software
python3 -m acoustic_imager.main
```

**Method 2: From acoustic_imager directory**
```bash
cd src/software/acoustic_imager
python3 main.py
```

### First Run

The application will start in **SIM mode** (simulated signals) by default, which doesn't require hardware. You'll see:
- Real-time beamforming heatmap
- Interactive frequency bar with bandpass controls
- Debug information overlay
- Control buttons for camera, source switching, and recording

---

## 📁 Repository Structure

```
Capstone_490_Software/
├── README.md                          # This file
├── src/
│   └── software/
│       └── acoustic_imager/           # Main application
│           ├── main.py                # Application entry point (~450 lines)
│           ├── config.py              # Configuration constants
│           ├── state.py               # Runtime state management
│           ├── custom_types.py        # Shared data types
│           │
│           ├── sources/               # Data source implementations
│           │   ├── sim_source.py     # Simulated signal generator
│           │   └── spi_source.py     # SPI hardware interface
│           │
│           ├── io/                    # Input/Output managers
│           │   ├── camera_manager.py # Camera capture (multi-backend)
│           │   └── spi_manager.py    # Low-level SPI communication
│           │
│           ├── dsp/                   # Digital Signal Processing
│           │   ├── beamforming.py    # MUSIC and ESPRIT algorithms
│           │   ├── heatmap.py        # Heatmap generation
│           │   └── bars.py           # Frequency/dB visualization
│           │
│           ├── spi/                   # SPI protocol layer
│           │   └── spi_protocol.py   # Frame format, validation, CRC
│           │
│           └── ui/                    # User Interface
│               ├── ui_components.py  # Buttons, menus, mouse handlers
│               └── video_recorder.py # Video recording via ffmpeg
│
└── docs/                              # Documentation (if applicable)
```

---

## Scientific Basis

### Array Geometry: Fermat's Spiral

We use a **Fermat spiral** layout inspired by phyllotaxis (sunflower seed packing):

**Why Fermat?**
- **Isotropic coverage**: Uniform angular resolution in all directions
- **Irregular spacing**: Smears aliasing into noise floor, improving MUSIC/ESPRIT convergence
- **Optimal packing**: Maximizes aperture for given number of elements

**Key Parameters:**
- 16 microphones
- Golden angle: ~137.5°
- Aperture diameter: ~8 cm
- Angular resolution: ~9° at 36 kHz

### Key Equations

**Angular Resolution:**
```
Δθ ≈ λ/D = c/(f·D)
```
Where: λ = wavelength, D = aperture diameter, c = speed of sound, f = frequency

**Spatial Aliasing Cutoff:**
```
f_alias = c/(2·d_min)
```
Where: d_min = minimum element spacing

**MUSIC Pseudospectrum:**
```
P_MUSIC(θ) = 1 / (a^H(θ) · E_n · E_n^H · a(θ))
```
Where: a(θ) = steering vector, E_n = noise subspace

### Example Performance (16 mics, 8 cm aperture, 36 kHz bandwidth)
- **Angular resolution**: ~9° at 36 kHz
- **Aliasing cutoff**: ~8.6 kHz
- **Higher frequencies**: Handled by MUSIC/ESPRIT subspace methods

---

## User Interface & Controls

### Mouse Controls
- **Left Click**: Interact with buttons and menu items
- **Drag**: Adjust bandpass filter range on frequency bar (green handles)

### Keyboard Shortcuts
- **Q** or **ESC**: Quit application

### UI Elements

**Bottom Control Bar:**
- **Camera**: Toggle camera overlay (ON/OFF/N/A)
- **Source**: Switch between SIM (simulated) and SPI (hardware) modes
- **DEBUG**: Toggle debug information overlay

**Top-Right Menu:**
- **30FPS / 60FPS / MAX**: Frame rate control
- **GAIN**: Gain mode toggle (LOW/HIGH)
- **SHOT**: Take screenshot (saved to `heatmap_captures/`)
- **REC**: Start/stop video recording
- **PAUSE**: Pause/resume recording

### Visualization Features
- **Heatmap**: Real-time beamforming visualization with camera overlay
- **Frequency Bar**: Shows signal power across frequency spectrum
- **Bandpass Filter**: Interactive draggable handles to select frequency range
- **dB Colorbar**: Color scale for heatmap intensity
- **Debug Overlay**: Frame count, FPS, source stats, throughput

---

## Configuration

Edit `src/software/acoustic_imager/config.py` to customize system parameters:

```python
# Display Settings
WIDTH = 1024
HEIGHT = 600
ALPHA = 0.7                    # Heatmap transparency

# Microphone Array
N_MICS = 16
SAMPLE_RATE_HZ = 100000        # 100 kHz sampling rate
SAMPLES_PER_CHANNEL = 512      # FFT size

# Physical Parameters
SPEED_SOUND = 343.0            # m/s at 20°C

# Visualization
REL_DB_MIN = -60.0             # Minimum dB for heatmap
REL_DB_MAX = 0.0               # Maximum dB for heatmap
F_DISPLAY_MAX = 45000.0        # Maximum frequency to display (Hz)

# Simulated Sources (SIM mode)
SIM_SOURCE_FREQS = [9000, 11000, 30000]     # Hz
SIM_SOURCE_ANGLES = [-35.0, 0.0, 40.0]      # degrees
SIM_SOURCE_AMPLS = [0.6, 1.0, 2.0]          # relative amplitudes
```

---

## Software Architecture

### Modular Design

The codebase has been refactored from a monolithic ~1760-line script into a clean, modular architecture:

**Benefits:**
- **Modularity**: Each module has a single, well-defined responsibility
- **Maintainability**: Easy to locate and modify specific functionality
- **Reusability**: Components can be used independently or in other projects
- **Testability**: Individual modules can be tested in isolation
- **Extensibility**: Simple to add new features, algorithms, or data sources

### Module Overview

| Module | Purpose | Key Components |
|--------|---------|----------------|
| **main.py** | Application orchestration | Main loop, event handling, initialization |
| **config.py** | Configuration | Constants, array geometry, parameters |
| **state.py** | Runtime state | UI state, bandpass settings, shared data |
| **sources/** | Data acquisition | SimSource, SPISource |
| **io/** | I/O management | CameraManager, SPIManager |
| **dsp/** | Signal processing | MUSIC, ESPRIT, heatmap generation |
| **spi/** | SPI protocol | Frame format, validation, CRC |
| **ui/** | User interface | Buttons, menus, video recording |

### Import Structure

All modules use absolute imports with the `acoustic_imager` package prefix:

```python
# Correct ✅
from acoustic_imager import config
from acoustic_imager.dsp.beamforming import music_spectrum
from acoustic_imager.sources.sim_source import SimSource

# Incorrect ❌
import config
from dsp.beamforming import music_spectrum
```

---

## Development

### Adding a New Data Source

1. Create a new file in `sources/`:
```python
# sources/my_source.py
from acoustic_imager.custom_types import LatestFrame, SourceStats

class MySource:
    def read_frame(self) -> LatestFrame:
        # Generate or acquire FFT data
        fft_data = ...  # shape: (N_MICS, N_BINS) complex64
        
        return LatestFrame(
            fft_data=fft_data,
            frame_id=self.frame_count,
            ok=True,
            stats=SourceStats(frames_ok=self.frame_count)
        )
```

2. Import and use in `main.py`:
```python
from acoustic_imager.sources.my_source import MySource
my_source = MySource()
```

### Adding a New Beamforming Algorithm

1. Add function to `dsp/beamforming.py`:
```python
def my_algorithm(
    R: np.ndarray,           # Covariance matrix
    angles: np.ndarray,      # Angle grid
    f_signal: float,         # Frequency
    n_sources: int,          # Number of sources
    x_coords: np.ndarray,    # Mic x positions
    y_coords: np.ndarray,    # Mic y positions
    speed_sound: float       # Speed of sound
) -> np.ndarray:
    # Return spectrum array (normalized to max=1)
    spectrum = ...
    return spectrum / (spectrum.max() + 1e-12)
```

2. Call from main loop in `main.py`:
```python
from acoustic_imager.dsp.beamforming import my_algorithm

spectrum = my_algorithm(R, config.ANGLES, f_sig, n_sources,
                       config.x_coords, config.y_coords, config.SPEED_SOUND)
```

### Testing

**Verify imports:**
```bash
cd src/software
python3 -c "from acoustic_imager import main; print('✅ All imports OK')"
```

**Run in SIM mode (no hardware required):**
```bash
cd src/software
python3 -m acoustic_imager.main
```

**Check module structure:**
```bash
cd src/software
python3 << 'EOF'
from acoustic_imager.sources.sim_source import SimSource
sim = SimSource()
frame = sim.read_frame()
print(f"✅ Generated frame {frame.frame_id}")
print(f"   Shape: {frame.fft_data.shape}")
print(f"   Dtype: {frame.fft_data.dtype}")
EOF
```

---

## Performance

### Typical Performance (Raspberry Pi 4)
- **SIM mode**: 60 FPS sustained
- **SPI mode**: 30-60 FPS (depends on data rate)
- **With camera overlay**: 30-45 FPS

### Optimization Tips
- Use **MAX FPS mode** only when needed (increases CPU usage)
- Disable camera if not required (saves ~10-15 FPS)
- Reduce `N_MICS` in config for faster processing (at cost of resolution)
- Check profiler output (printed every 60 frames) to identify bottlenecks

### Profiler Output Example
```
ms avg | read=0.52 heat=8.34 bg=2.15 blend=1.87 bars=0.94 ui=0.67 imshow=3.21 waitKey=0.15 total=18.23
```

---

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'acoustic_imager'`

**Solution:**
```bash
# Option 1: Run from correct directory
cd src/software
python3 -m acoustic_imager.main

# Option 2: Add to PYTHONPATH
export PYTHONPATH="/path/to/Capstone_490_Software/src/software:$PYTHONPATH"
```

### Camera Not Detected

**Problem:** Camera not working or showing "N/A"

**Solutions:**
- Check permissions: `sudo usermod -a -G video $USER`
- Verify camera: `libcamera-hello` or `v4l2-ctl --list-devices`
- Try different backends in `config.py` (Picamera2, GStreamer, V4L2)
- On macOS: Grant camera permissions in System Preferences

### SPI Not Working

**Problem:** SPI mode shows errors or no data

**Solutions:**
- Check device exists: `ls -l /dev/spidev*`
- Add user to spi group: `sudo usermod -a -G spi $USER`
- Enable SPI on Raspberry Pi: `sudo raspi-config` → Interface Options → SPI
- For loopback testing: Connect MOSI to MISO physically

### Low FPS / Performance Issues

**Solutions:**
- Check profiler output to identify bottleneck stage
- Reduce resolution in `config.py` (WIDTH, HEIGHT)
- Disable camera overlay
- Use 30 FPS mode instead of 60 or MAX
- Close other applications

### Missing Dependencies

**Problem:** Import errors for `numpy`, `cv2`, etc.

**Solution:**
```bash
pip3 install numpy opencv-python
pip3 install spidev picamera2  # Optional, for hardware
```

---

## References

### Academic Papers
- Schmidt, R. (1986). "Multiple Emitter Location and Signal Parameter Estimation" (MUSIC algorithm)
- Roy, R., & Kailath, T. (1989). "ESPRIT—Estimation of Signal Parameters via Rotational Invariance Techniques"
- Van Trees, H. L. (2002). "Detection, Estimation, and Modulation Theory" (Array signal processing)

### Industry Benchmarks
- Fluke ii900 Acoustic Imager (industrial reference system)

### Phyllotaxis & Geometry
- Vogel, H. (1979). "A better way to construct the sunflower head" (Fermat spiral)

---

## Project Roadmap

### Phase I (Fall Semester) ✅
- ✅ Literature review and requirements analysis
- ✅ Microphone selection and array design
- ✅ PCB and housing design
- ✅ Python DSP prototypes
- ✅ Simulation tools

### Phase II (Winter Semester) ✅
- ✅ Software architecture refactoring
- ✅ MUSIC/ESPRIT implementation
- ✅ Real-time visualization with interactive controls
- ✅ Camera overlay integration
- ✅ Video recording functionality
- 🔄 STM32 firmware development (in progress)
- 🔄 Raspberry Pi integration (in progress)
- 🔄 Final hardware validation (in progress)

### Future Enhancements
- Unit tests for DSP algorithms
- Configuration file support (YAML/JSON)
- Plugin system for custom algorithms
- API documentation (Sphinx)
- Mobile app integration

---

**Version**: 2.0 (Refactored Architecture)  
**Last Updated**: February 15, 2026  
**Status**: ✅ Software Complete | 🔄 Hardware Integration In Progress

