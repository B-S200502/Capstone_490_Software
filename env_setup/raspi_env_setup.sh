#!/bin/bash
set -e

echo "============================================"
echo " Acoustic Imager Base Setup (Raspberry Pi)"
echo "============================================"

# ---------- 1. Update and Upgrade ----------
echo "[1/5] Updating system packages..."
sudo apt update
sudo apt upgrade -y


# ---------- 2. Install Python & Core Packages ----------
echo "[2/5] Installing Python, pip, venv, and build tools..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential


# ---------- 3. Install DSP + OpenCV Dependencies ----------
echo "[3/5] Installing NumPy, SciPy, OpenCV..."
sudo apt install -y \
    python3-numpy \
    python3-scipy \
    python3-opencv


# ---------- 4. Install Camera Support ----------
echo "[4/5] Installing libcamera tools..."
sudo apt install -y \
    libcamera0 \
    libcamera-tools \
    libcamera-apps \
    python3-libcamera \
    python3-kms++


# ---------- 5. Install Git ----------
echo "[5/5] Installing Git..."
sudo apt install -y git


echo "============================================"
echo " Base Setup Complete!"
echo ""
echo "Next Steps:"
echo ""
echo "1) Clone the repository manually:"
echo "   git clone https://github.com/B-S200502/Capstone_490_Software.git"
echo ""
echo "2) Create virtual environment:"
echo "   cd Capstone_490_Software"
echo "   python3 -m venv venv"
echo "   source venv/bin/activate"
echo "   pip install -r requirements.txt"
echo ""
echo "============================================"
