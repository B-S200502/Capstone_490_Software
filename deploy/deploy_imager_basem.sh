#!/usr/bin/env bash
set -euo pipefail

# ============================
# Acoustic Imager Service Setup
# Supports:
#   -dev     Enable DEV mode
#   -status  Show DEV mode status
# ============================

# ---- CONFIG ----
ACOUSTIC_USER="basempi"
ACOUSTIC_GROUP="basempi"
PROJECT_DIR="/home/basempi/Capstone_490_Software"
PYTHON_BIN="/usr/bin/python3"
APP_ENTRY="/home/basempi/Capstone_490_Software/src/software/acoustic_imager/main.py"

TMUX_SESSION="acoustic_ui"
SERVICE_NAME="acoustic-ui.service"
DEV_MODE_FILE="/home/basempi/DEV_MODE"
RUN_WRAPPER="/usr/local/bin/acoustic-ui-run"

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

# If only asking for status, don't install anything
if [[ "$SHOW_STATUS" -eq 1 ]]; then
    show_status
fi

# ----------------------------
# If service exists already,
# just toggle DEV mode
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
# Otherwise full install
# ----------------------------
if [[ "$EUID" -ne 0 ]]; then
    echo "Please run with sudo for first-time install."
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

# Run from the package-root so imports work
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
