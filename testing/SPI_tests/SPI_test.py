import spidev
import time
import random

# Open SPI bus 0, device 0 (CE0)
spi = spidev.SpiDev()
spi.open(0, 0)

# Configure SPI
spi.max_speed_hz = 1000000   # 1 MHz (start safe)
spi.mode = 0                 # SPI mode 0 (CPOL=0, CPHA=0)
spi.bits_per_word = 8

print("SPI loopback test started...")

try:
    while True:
        tx_data = [random.randint(0, 255) for _ in range(16)]  # 16 random bytes

        rx_data = spi.xfer2(tx_data)

        print("TX:", [hex(x) for x in tx_data])
        print("RX:", [hex(x) for x in rx_data])
        print("------------------------")

        time.sleep(1)

except KeyboardInterrupt:
    print("Exiting...")
    spi.close()
