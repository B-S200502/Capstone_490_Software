import spidev
import time
import zlib
import sys
import struct
import csv
import resource
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# If you don't need FFTFrame for this sweep test, you can remove these two lines.
sys.path.append(str(Path(__file__).resolve().parents[1] / "dataframe"))
from fftframe import FFTFrame  # noqa: F401

# ---------------------------
# Config
# ---------------------------
N_MICS = 16
SAMPLE_RATE_HZ = 100000
SAMPLES_PER_CHANNEL = 1024

# rFFT bins for real FFT: N/2 + 1
N_BINS = SAMPLES_PER_CHANNEL // 2 + 1  # 513

FPS_TARGET = 30
CHUNK = 4096
DURATION_S = 5  # per-speed test duration (seconds)

LIMIT_30_FPS_MODE = False   # False = unthrottled
CRC_EVERY_N = 1             # set to 1 during sweep to find corruption cliff

SWEEP_MHZ_START = 30
SWEEP_MHZ_END = 120
SWEEP_MHZ_STEP = 10

# ---------------------------
# Frame format
# ---------------------------
MAGIC_START = 0x46524654  # arbitrary stable tag
MAGIC_END   = 0x454E4421  # 'END!'

VERSION = 1

# Little-endian header:
# magic(u32), ver(u16), hdr_len(u16),
# seq(u32),
# mic_count(u16), fft_size(u16),
# sample_rate(u32),
# bins(u16), reserved(u16),
# payload_len(u32)
HEADER_FMT = "<IHH I HH I HH I"
HEADER_LEN = struct.calcsize(HEADER_FMT)

# Trailer: crc32(u32), end_magic(u32)
TRAILER_FMT = "<II"
TRAILER_LEN = struct.calcsize(TRAILER_FMT)

PAYLOAD_LEN = N_MICS * N_BINS * 4  # float32
FRAME_BYTES = HEADER_LEN + PAYLOAD_LEN + TRAILER_LEN


def make_payload(seq: int) -> bytes:
    """
    Deterministic payload shaped like (N_MICS, N_BINS) float32.
    This emulates '513 bins * 4 bytes * 16 channels'.
    """
    base = (seq % 1024) * 0.01
    arr = np.empty((N_MICS, N_BINS), dtype=np.float32)
    bins_vec = (np.arange(N_BINS, dtype=np.float32) * 1e-4)
    for ch in range(N_MICS):
        arr[ch, :] = base + ch + bins_vec
    return arr.tobytes(order="C")


def make_frame(seq: int) -> bytearray:
    payload = make_payload(seq)

    header = struct.pack(
        HEADER_FMT,
        MAGIC_START,
        VERSION,
        HEADER_LEN,
        seq,
        N_MICS,
        SAMPLES_PER_CHANNEL,
        SAMPLE_RATE_HZ,
        N_BINS,
        0,  # reserved
        len(payload),
    )

    crc = zlib.crc32(header)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    trailer = struct.pack(TRAILER_FMT, crc, MAGIC_END)

    return bytearray(header + payload + trailer)


def xfer_bytes(spi: spidev.SpiDev, tx: memoryview) -> bytearray:
    rx = bytearray(len(tx))
    offset = 0
    while offset < len(tx):
        end = min(offset + CHUNK, len(tx))
        chunk = tx[offset:end]
        r = spi.xfer2(chunk)  # works on your Pi; avoids list() conversion
        rx[offset:end] = bytes(r)
        offset = end
    return rx


def crc_validate_frame(buf: bytes) -> bool:
    """Full validation: framing + CRC."""
    if len(buf) != FRAME_BYTES:
        return False

    try:
        (magic, ver, hdr_len, seq, mic, fft_size, fs, bins, _res, pay_len) = struct.unpack_from(
            HEADER_FMT, buf, 0
        )
    except struct.error:
        return False

    if magic != MAGIC_START or ver != VERSION or hdr_len != HEADER_LEN:
        return False
    if mic != N_MICS or fft_size != SAMPLES_PER_CHANNEL or fs != SAMPLE_RATE_HZ or bins != N_BINS:
        return False
    if pay_len != PAYLOAD_LEN:
        return False

    trailer_off = HEADER_LEN + PAYLOAD_LEN
    (crc_rx, magic_end) = struct.unpack_from(TRAILER_FMT, buf, trailer_off)
    if magic_end != MAGIC_END:
        return False

    header = buf[:HEADER_LEN]
    payload = buf[HEADER_LEN:HEADER_LEN + PAYLOAD_LEN]
    crc = zlib.crc32(header)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return crc == crc_rx


def framing_validate_frame(buf: bytes) -> tuple[bool, str]:
    """Fast framing validation only (no CRC)."""
    if len(buf) != FRAME_BYTES:
        return False, "len"

    (magic,) = struct.unpack_from("<I", buf, 0)
    if magic != MAGIC_START:
        return False, "magic_start"

    try:
        (magic, ver, hdr_len, seq, mic, fft_size, fs, bins, _res, pay_len) = struct.unpack_from(
            HEADER_FMT, buf, 0
        )
    except struct.error:
        return False, "hdr_unpack"

    if ver != VERSION or hdr_len != HEADER_LEN:
        return False, "hdr_fields"
    if mic != N_MICS or fft_size != SAMPLES_PER_CHANNEL or fs != SAMPLE_RATE_HZ or bins != N_BINS:
        return False, "cfg_mismatch"
    if pay_len != PAYLOAD_LEN:
        return False, "pay_len"

    (_, magic_end) = struct.unpack_from(TRAILER_FMT, buf, HEADER_LEN + PAYLOAD_LEN)
    if magic_end != MAGIC_END:
        return False, "magic_end"

    return True, "ok"


def get_cpu_time_seconds():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime, ru.ru_stime


def run_test(spi_hz: int) -> dict:
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.mode = 0
    spi.bits_per_word = 8
    spi.max_speed_hz = spi_hz
    sclk_reported_hz = spi.max_speed_hz  # snapshot

    frames = 0
    bad_loopback = 0
    bad_parse = 0
    crc_checked = 0
    crc_skipped = 0
    bad_crc = 0
    bytes_total = 0
    next_deadline = time.perf_counter()

    t_start = time.perf_counter()
    t_end = t_start + DURATION_S
    u0, s0 = get_cpu_time_seconds()

    try:
        seq = 0
        while True:
            now = time.perf_counter()
            if now >= t_end:
                break

            if LIMIT_30_FPS_MODE:
                next_deadline += 1.0 / FPS_TARGET
                if now < next_deadline:
                    time.sleep(max(0.0, next_deadline - now - 0.0003))

            tx = make_frame(seq)
            rx = xfer_bytes(spi, memoryview(tx))

            if rx != tx:
                bad_loopback += 1

            ok, why = framing_validate_frame(rx)
            if not ok:
                bad_parse += 1
                crc_skipped += 1
            else:
                if (seq % CRC_EVERY_N) == 0:
                    crc_checked += 1
                    if not crc_validate_frame(rx):
                        bad_crc += 1

            frames += 1
            bytes_total += FRAME_BYTES
            seq += 1

    finally:
        spi.close()

    elapsed = time.perf_counter() - t_start
    u1, s1 = get_cpu_time_seconds()
    proc_cpu = (u1 - u0) + (s1 - s0)
    cpu_pct_1core = 100.0 * proc_cpu / elapsed

    mb_s = bytes_total / elapsed / 1e6
    mbps = (bytes_total * 8) / elapsed / 1e6
    fps = frames / elapsed

    return {
        "spi_hz_req": spi_hz,
        "sclk_hz_rep": sclk_reported_hz,
        "elapsed_s": elapsed,
        "frames": frames,
        "fps": fps,
        "mb_s": mb_s,
        "mbps": mbps,
        "bad_loopback": bad_loopback,
        "bad_parse": bad_parse,
        "crc_checked": crc_checked,
        "crc_skipped": crc_skipped,
        "bad_crc": bad_crc,
        "cpu_pct_1core": cpu_pct_1core,
    }


def main():
    speeds_mhz = list(range(SWEEP_MHZ_START, SWEEP_MHZ_END + 1, SWEEP_MHZ_STEP))
    results = []

    print(f"Frame bytes: {FRAME_BYTES} | Chunk: {CHUNK} | Duration/pt: {DURATION_S}s")
    print(f"Mode: {'LIMITED' if LIMIT_30_FPS_MODE else 'UNTHROTTLED'} | CRC every N: {CRC_EVERY_N}")
    print("Sweep:", speeds_mhz, "MHz\n")

    for mhz in speeds_mhz:
        hz = mhz * 1_000_000
        r = run_test(hz)
        results.append(r)

        print(
            f"{mhz:>3} MHz -> {r['mbps']:>6.2f} Mb/s | fps {r['fps']:>6.1f} | "
            f"loopBad {r['bad_loopback']:>4} | parseBad {r['bad_parse']:>4} | "
            f"crcBad {r['bad_crc']:>3} (chk {r['crc_checked']}, skip {r['crc_skipped']})"
        )

    # Summary table
    print("\nSUMMARY")
    print("MHz  Mb/s    FPS   loopBad  parseBad  crcBad  crcChk  crcSkip  CPU%")
    print("--------------------------------------------------------------------")
    for r in results:
        mhz = r["spi_hz_req"] / 1e6
        print(
            f"{mhz:>3.0f}  {r['mbps']:>6.1f}  {r['fps']:>6.1f}  "
            f"{r['bad_loopback']:>7}  {r['bad_parse']:>8}  {r['bad_crc']:>6}  "
            f"{r['crc_checked']:>6}  {r['crc_skipped']:>7}  {r['cpu_pct_1core']:>4.1f}"
        )

    # Save CSV
    out_csv = "spi_sweep_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote: {out_csv}")

    # Plot throughput vs MHz
    mhz = [r["spi_hz_req"] / 1e6 for r in results]
    mbps = [r["mbps"] for r in results]
    loop_bad = [r["bad_loopback"] for r in results]
    parse_bad = [r["bad_parse"] for r in results]
    crc_bad = [r["bad_crc"] for r in results]

    plt.figure()
    plt.plot(mhz, mbps, marker="o")
    plt.xlabel("SPI clock (MHz)")
    plt.ylabel("Throughput (Mb/s)")
    plt.title("SPI loopback throughput vs clock")
    plt.grid(True)

    plt.figure()
    plt.plot(mhz, loop_bad, marker="o", label="Loopback mismatches")
    plt.plot(mhz, parse_bad, marker="o", label="Framing fails")
    plt.plot(mhz, crc_bad, marker="o", label="CRC fails (sampled)")
    plt.xlabel("SPI clock (MHz)")
    plt.ylabel("Count")
    plt.title("SPI loopback errors vs clock")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
