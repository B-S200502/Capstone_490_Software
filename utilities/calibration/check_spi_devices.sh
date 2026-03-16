#!/bin/bash
# Quick SPI diagnostic: config, device nodes, kernel modules.
# Run: bash utilities/calibration/check_spi_devices.sh

echo "=== Boot config (SPI enabled?) ==="
for f in /boot/config.txt /boot/firmware/config.txt; do
  if [[ -f "$f" ]]; then
    echo "--- $f ---"
    grep -n -E "spi|dtparam" "$f" 2>/dev/null || true
  fi
done

echo ""
echo "=== SPI device nodes ==="
ls -la /dev/spidev* 2>/dev/null || echo "No /dev/spidev* found"

echo ""
echo "=== Kernel SPI modules ==="
lsmod | grep -E "spi|spidev" || echo "No spi/spidev modules loaded"

echo ""
echo "=== dmesg (SPI) ==="
dmesg 2>/dev/null | grep -i spi | tail -20
