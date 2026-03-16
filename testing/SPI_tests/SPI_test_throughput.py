import spidev
import time
import os
import zlib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "dataframe"))
from fftframe import FFTFrame

N_MICS = 16
SAMPLE_RATE_HZ = 100000
SAMPLES_PER_CHANNEL = 1024


# Target payload: 513 bins * 4 bytes * 16 channels
FRAME_BYTES = 513 * 4 * 16          # 32832 bytes
FPS_TARGET = 30
CHUNK = 4096                         # 4096 or 8192 are good starting points
SPI_HZ = 16_000_000                  # try 10_000_000 first, then 12/16/20 MHz

DURATION_S = 100                     # how long to run the test

def make_frame(seq: int) -> bytearray:
    """
    Create deterministic frame data so loopback can be verified.
    Mix seq into the frame so it's not constant.
    """
    b = bytearray(FRAME_BYTES)
    # Fill with a repeating pattern + seq
    # (Fast enough, and deterministic)
    v = seq & 0xFF
    for i in range(FRAME_BYTES):
        b[i] = (i + v) & 0xFF
    return b

def xfer_bytes(spi: spidev.SpiDev, tx: memoryview) -> bytearray:
    """
    Transfer tx bytes over SPI in CHUNK blocks; return rx bytes.
    Using list(tx_slice) is simplest/reliable across spidev versions.
    """
    rx = bytearray(len(tx))
    offset = 0
    while offset < len(tx):
        end = min(offset + CHUNK, len(tx))
        tx_slice = tx[offset:end]
        # xfer2 wants a list of ints
        r = spi.xfer2(list(tx_slice))
        rx[offset:end] = bytes(r)
        offset = end
    return rx

def main():
    spi = spidev.SpiDev()
    spi.open(0, 0)                   # /dev/spidev0.0
    spi.mode = 0
    spi.bits_per_word = 8
    spi.max_speed_hz = SPI_HZ

    print(f"SPI Hz: {SPI_HZ:,}")
    print(f"Frame bytes: {FRAME_BYTES}  | Target FPS: {FPS_TARGET}  | Target rate: {FRAME_BYTES*FPS_TARGET/1e6:.3f} MB/s")
    print(f"Chunk size: {CHUNK}")
    print("Running... (no printing in the loop)")

    t_start = time.perf_counter()
    t_end = t_start + DURATION_S

    frames = 0
    bad = 0
    bytes_total = 0

    next_deadline = t_start

    try:
        seq = 0
        while True:
            now = time.perf_counter()
            if now >= t_end:
                break

            # Pace to 30 fps
            next_deadline += 1.0 / FPS_TARGET
            if now < next_deadline:
                # sleep a bit, but don't oversleep too hard
                time.sleep(max(0.0, next_deadline - now - 0.0003))

            tx = make_frame(seq)
            tx_mv = memoryview(tx)

            rx = xfer_bytes(spi, tx_mv)

            # Verify occasionally (every frame here; you can reduce if needed)
            if rx != tx:
                bad += 1

            frames += 1
            bytes_total += FRAME_BYTES
            seq += 1

    finally:
        spi.close()

    elapsed = time.perf_counter() - t_start
    mb_s = bytes_total / elapsed / 1e6
    mbps = (bytes_total * 8) / elapsed / 1e6
    fps = frames / elapsed

    print("\nRESULTS")
    print(f"Elapsed: {elapsed:.3f} s")
    print(f"Frames:  {frames}  | FPS achieved: {fps:.2f}")
    print(f"Data:    {bytes_total/1e6:.3f} MB  | Throughput: {mb_s:.3f} MB/s ({mbps:.3f} Mb/s)")
    print(f"Loopback mismatches: {bad}")

if __name__ == "__main__":
    main()
