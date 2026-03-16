#!/usr/bin/env bash
set -euo pipefail

THEME_NAME="mytheme"
THEME_SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THEME_DEST_DIR="/usr/share/plymouth/themes/${THEME_NAME}"

echo "Installing Acoustic Imager splash theme..."

# ---- 1) Ensure required files exist in repo ----
for f in "splash.png" "${THEME_NAME}.plymouth" "${THEME_NAME}.script"; do
  if [[ ! -f "${THEME_SRC_DIR}/${f}" ]]; then
    echo "ERROR: Missing ${f} in ${THEME_SRC_DIR}"
    exit 1
  fi
done

# ---- 2) Ensure plymouth is installed ----
if ! dpkg -s plymouth >/dev/null 2>&1; then
  echo "Installing plymouth..."
  sudo apt update
  sudo apt install -y plymouth plymouth-themes
fi

# ---- 3) Copy files into system theme directory ----
sudo mkdir -p "$THEME_DEST_DIR"
sudo cp "${THEME_SRC_DIR}/splash.png"        "$THEME_DEST_DIR/"
sudo cp "${THEME_SRC_DIR}/${THEME_NAME}.plymouth" "$THEME_DEST_DIR/"
sudo cp "${THEME_SRC_DIR}/${THEME_NAME}.script"   "$THEME_DEST_DIR/"

# Some Plymouth builds look for .theme specifically
sudo cp "${THEME_DEST_DIR}/${THEME_NAME}.plymouth" "${THEME_DEST_DIR}/${THEME_NAME}.theme"

# Ensure readable permissions
sudo chmod 644 "${THEME_DEST_DIR}/splash.png" \
               "${THEME_DEST_DIR}/${THEME_NAME}.plymouth" \
               "${THEME_DEST_DIR}/${THEME_NAME}.theme" \
               "${THEME_DEST_DIR}/${THEME_NAME}.script"

# ---- 4) Reset broken alternatives group if it exists (prevents that warning) ----
sudo update-alternatives --remove-all default.plymouth >/dev/null 2>&1 || true

# ---- 5) Set theme using the correct tool for Raspberry Pi OS ----
sudo plymouth-set-default-theme -R "$THEME_NAME"

echo "Splash theme installed successfully."
echo "Tip: test without reboot using:"
echo "  sudo pkill plymouthd || true; sudo plymouthd; sudo plymouth --show-splash; sleep 3; sudo plymouth quit"
echo "Reboot to see changes."