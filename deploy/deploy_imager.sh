#!/usr/bin/env bash
set -euo pipefail

# ============================
# Acoustic Imager: Full deploy (env setup + service)
# - First run: system packages + Python/libcamera/git, then service
# - Later runs: service only (same device)
# Supports: -dev, -status, --clean
# ============================

# ---- CONFIG (device name and paths detected from current/sudo user) ----
ACOUSTIC_USER="${SUDO_USER:-$(whoami)}"
ACOUSTIC_GROUP="$ACOUSTIC_USER"
PROJECT_DIR="/home/$ACOUSTIC_USER/Capstone_490_Software"
PYTHON_BIN="/usr/bin/python3"
TMUX_SESSION="acoustic_ui"
SERVICE_NAME="acoustic-ui.service"
DEV_MODE_FILE="/home/$ACOUSTIC_USER/DEV_MODE"
RUN_WRAPPER="/usr/local/bin/acoustic-ui-run"
ENV_SETUP_SENTINEL="/home/$ACOUSTIC_USER/.acoustic_imager_env_setup_done"
CONFIG_DIR="/home/$ACOUSTIC_USER/.config/acoustic-imager"
WIFI_GEO_API_KEY_FILE="$CONFIG_DIR/wifi_geo_api_key"

ENABLE_DEV=0
SHOW_STATUS=0
CLEAN_REINSTALL=0

# ----------------------------
# Parse flags
# ----------------------------
for arg in "$@"; do
    case "$arg" in
    -dev)
        ENABLE_DEV=1
        ;;
    -status)
        SHOW_STATUS=1
        ;;
    --clean)
        CLEAN_REINSTALL=1
        ;;
    *)
        echo "Unknown option: $arg"
        exit 1
        ;;
    esac
done

# ----------------------------
# DEV MODE HANDLER
# ----------------------------
handle_dev_mode() {
    if [[ "$ENABLE_DEV" -eq 1 ]]; then
        echo "Enabling DEV mode..."
        touch "$DEV_MODE_FILE"
    else
        echo "Disabling DEV mode..."
        rm -f "$DEV_MODE_FILE"
    fi

    sudo systemctl restart "$SERVICE_NAME" || true
}

show_status() {
    echo
    echo "DEV MODE STATUS:"
    if [[ -f "$DEV_MODE_FILE" ]]; then
        echo "  ENABLED"
    else
        echo "  DISABLED"
    fi

    echo
    sudo systemctl status "$SERVICE_NAME" --no-pager
    exit 0
}

# If only asking for status, don't install or run env setup
if [[ "$SHOW_STATUS" -eq 1 ]]; then
    show_status
fi

# ----------------------------
# Detect if env is already set up (e.g. manual install or older script never created sentinel)
# ----------------------------
env_already_set_up() {
    command -v python3 &>/dev/null || return 1
    python3 -c "import cv2" 2>/dev/null || return 1
    dpkg -s python3-libcamera &>/dev/null && return 0
    dpkg -s rpicam-apps &>/dev/null && return 0
    dpkg -s libcamera0.7 &>/dev/null && return 0
    dpkg -s libcamera0 &>/dev/null && return 0
    return 1
}

# ----------------------------
# First-time: environment setup (once per device)
# ----------------------------
if [[ ! -f "$ENV_SETUP_SENTINEL" ]]; then
    if env_already_set_up; then
        echo "Environment already set up (detected). Creating sentinel and skipping base setup."
        if [[ -n "${SUDO_USER:-}" ]]; then
            sudo -u "$ACOUSTIC_USER" touch "$ENV_SETUP_SENTINEL"
        else
            touch "$ENV_SETUP_SENTINEL"
        fi
    else
        echo "First-time deploy: running environment setup for this device..."
        echo "============================================"
        echo " Acoustic Imager Base Setup (Raspberry Pi)"
        echo "============================================"

        echo "[1/5] Updating system packages..."
        sudo apt update
        sudo apt upgrade -y

        echo "[2/5] Installing Python, pip, venv, and build tools..."
        sudo apt install -y \
            python3 \
            python3-pip \
            python3-venv \
            python3-dev \
            build-essential

        echo "[3/5] Installing NumPy, SciPy, OpenCV..."
        sudo apt install -y \
            python3-numpy \
            python3-scipy \
            python3-opencv

        echo "[4/5] Installing libcamera tools..."
        if ! sudo apt install -y libcamera0.7 rpicam-apps python3-libcamera python3-kms++ 2>/dev/null; then
            sudo apt install -y \
                libcamera0 \
                libcamera-tools \
                libcamera-apps \
                python3-libcamera \
                python3-kms++
        fi

        echo "[5/5] Installing Git..."
        sudo apt install -y git

        if [[ -n "${SUDO_USER:-}" ]]; then
            sudo -u "$ACOUSTIC_USER" touch "$ENV_SETUP_SENTINEL"
        else
            touch "$ENV_SETUP_SENTINEL"
        fi
        echo "Environment setup completed (sentinel: $ENV_SETUP_SENTINEL)."
        echo "============================================"
    fi
else
    echo "Environment already set up for this device. Skipping base setup."
fi

# ----------------------------
# Google Geolocation API key (prompt if missing — used for Wi-Fi location on radar map)
# ----------------------------
if [[ ! -f "$WIFI_GEO_API_KEY_FILE" ]] || [[ ! -s "$WIFI_GEO_API_KEY_FILE" ]]; then
    echo
    echo "Google Geolocation API key not set (used for Wi-Fi location on the radar map)."
    echo "Enter your API key (or press Enter to skip):"
    read -r USER_API_KEY || true
    if [[ -n "${USER_API_KEY:-}" ]]; then
        if [[ "$EUID" -eq 0 ]] && [[ -n "${SUDO_USER:-}" ]]; then
            sudo -u "$ACOUSTIC_USER" mkdir -p "$CONFIG_DIR"
            echo -n "$USER_API_KEY" | sudo -u "$ACOUSTIC_USER" tee "$WIFI_GEO_API_KEY_FILE" > /dev/null
        else
            mkdir -p "$CONFIG_DIR"
            echo -n "$USER_API_KEY" > "$WIFI_GEO_API_KEY_FILE"
        fi
        echo "API key saved to $WIFI_GEO_API_KEY_FILE"
    else
        echo "Skipped. You can add a key later to $WIFI_GEO_API_KEY_FILE"
    fi
fi

# ----------------------------
# If service exists already,
# just toggle DEV mode (unless --clean)
# ----------------------------
if [[ -f "/etc/systemd/system/$SERVICE_NAME" ]]; then
    if [[ "$CLEAN_REINSTALL" -eq 1 ]]; then
        echo "Clean reinstall requested (--clean)."

        echo "Stopping service (if running)..."
        sudo systemctl disable --now "$SERVICE_NAME" 2>/dev/null || true

        echo "Killing old tmux session (if any)..."
        /usr/bin/tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

        echo "Removing old unit + wrapper..."
        sudo rm -f "/etc/systemd/system/$SERVICE_NAME"
        sudo rm -f "$RUN_WRAPPER"

        echo "Clearing old log..."
        sudo rm -f /tmp/acoustic_ui.log

        echo "Reloading systemd..."
        sudo systemctl daemon-reload

        echo "Old service + session fully removed."
    else
        echo "Service already installed."
        handle_dev_mode
        exit 0
    fi
fi

# ----------------------------
# Full install: need root
# ----------------------------
if [[ "$EUID" -ne 0 ]]; then
    echo "Please run with sudo for first-time service install."
    exit 1
fi

echo "Installing Acoustic UI service..."

if ! command -v tmux >/dev/null 2>&1; then
    echo "Installing tmux..."
    apt-get update -y
    apt-get install -y tmux
else
    echo "tmux already installed."
fi

# ----------------------------
# Create runtime wrapper
# ----------------------------
cat > "$RUN_WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

DEV_MODE_FILE="$DEV_MODE_FILE"

if [[ -f "\$DEV_MODE_FILE" ]]; then
  echo "[acoustic-ui] DEV_MODE enabled. Not starting UI."
  exit 0
fi

cd "$PROJECT_DIR/src/software"

if /usr/bin/tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "[acoustic-ui] Session already running."
else
  exec /usr/bin/tmux new-session -d -s "$TMUX_SESSION" \
    "$PYTHON_BIN -u -m acoustic_imager.main 2>&1 | /usr/bin/systemd-cat -t acoustic-ui"
fi
EOF

chmod 0755 "$RUN_WRAPPER"

# ----------------------------
# Create systemd service
# ----------------------------
cat > "/etc/systemd/system/$SERVICE_NAME" <<EOF
[Unit]
Description=Acoustic Imager UI (tmux managed)
After=display-manager.service
Wants=display-manager.service

[Service]
Type=oneshot
User=$ACOUSTIC_USER
Group=$ACOUSTIC_GROUP
WorkingDirectory=$PROJECT_DIR
RemainAfterExit=yes
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/$ACOUSTIC_USER/.Xauthority
ExecStart=$RUN_WRAPPER
ExecStop=/usr/bin/tmux kill-session -t $TMUX_SESSION
Restart=on-failure
RestartSec=1
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=graphical.target
EOF

# ----------------------------
# Activate service
# ----------------------------
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl start "$SERVICE_NAME"

echo "Installation complete."

handle_dev_mode

echo
echo "Setup complete."
echo "Reboot to verify auto-start:"
echo "  sudo reboot"
