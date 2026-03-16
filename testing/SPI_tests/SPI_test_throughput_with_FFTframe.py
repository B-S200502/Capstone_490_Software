import spidev
import time
import zlib
import sys
import struct
import os
import resource
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "dataframe"))
from fftframe import FFTFrame

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
SPI_HZ = 100_000_000
DURATION_S = 10

LIMIT_30_FPS_MODE = False
CRC_EVERY_N = 10


# ---------------------------
# Frame format (example)
# ---------------------------
MAGIC_START = 0x46524654  # 'TF RF' (just a tag) - pick anything stable
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
    # Create stable-but-changing values based on seq
    # (fast to generate; no trig)
    base = (seq % 1024) * 0.01
    arr = np.empty((N_MICS, N_BINS), dtype=np.float32)
    for ch in range(N_MICS):
        arr[ch, :] = base + ch + (np.arange(N_BINS, dtype=np.float32) * 1e-4)
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

    # CRC over header + payload (common embedded pattern)
    crc = zlib.crc32(header)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF

    trailer = struct.pack(TRAILER_FMT, crc, MAGIC_END)

    return bytearray(header + payload + trailer)


def xfer_bytes(spi: spidev.SpiDev, tx: memoryview) -> bytearray:
    rx = bytearray(len(tx))
    offset = 0
    while offset < len(tx):
        end = min(offset + CHUNK, len(tx))
        chunk = tx[offset:end]           # bytes/bytearray slice
        r = spi.xfer2(chunk)
        rx[offset:end] = bytes(r)
        offset = end
    return rx


def crc_validate_frame(buf: bytes) -> bool:
    """Basic check: header magic, lengths, end magic, crc."""
    if len(buf) != FRAME_BYTES:
        return False

    # Parse header
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

    payload_off = HEADER_LEN
    trailer_off = HEADER_LEN + PAYLOAD_LEN

    (crc_rx, magic_end) = struct.unpack_from(TRAILER_FMT, buf, trailer_off)
    if magic_end != MAGIC_END:
        return False

    header = buf[:HEADER_LEN]
    payload = buf[payload_off:payload_off + PAYLOAD_LEN]
    crc = zlib.crc32(header)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF

    return crc == crc_rx


def fast_validate_frame(buf: bytes) -> tuple[bool, str]:
    if len(buf) != FRAME_BYTES:
        return False, "len"

    # header magic quick
    (magic,) = struct.unpack_from("<I", buf, 0)
    if magic != MAGIC_START:
        return False, "magic_start"

    # unpack header once
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

    # footer magic quick
    (_, magic_end) = struct.unpack_from(TRAILER_FMT, buf, HEADER_LEN + PAYLOAD_LEN)
    if magic_end != MAGIC_END:
        return False, "magic_end"

    return True, "ok"


def get_mem_stats_linux():
    """Return (rss_mb, mem_total_mb, mem_avail_mb) using /proc on Linux."""
    rss_kb = 0
    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
                break

    mem_total_kb = mem_avail_kb = 0
    with open("/proc/meminfo", "r") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_avail_kb = int(line.split()[1])

    return rss_kb / 1024.0, mem_total_kb / 1024.0, mem_avail_kb / 1024.0


def get_cpu_time_seconds():
    """Return (user_cpu_s, sys_cpu_s) consumed by this process."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime, ru.ru_stime


def print_kv_table(title: str, rows, col1_w=24):
    """
    rows: list of (key, value_str)
    """
    print(f"\n{title}")
    print("-" * (col1_w + 2 + 40))
    for k, v in rows:
        print(f"{k:<{col1_w}} : {v}")


def main():
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.mode = 0
    spi.bits_per_word = 8
    spi.max_speed_hz = SPI_HZ
    sclk_reported_hz = spi.max_speed_hz


    print(f"SPI Hz: {SPI_HZ:,}")
    print(f"Frame bytes (with hdr+crc+footer): {FRAME_BYTES}")
    # print(f"Target FPS: {FPS_TARGET}  | Target rate: {FRAME_BYTES*FPS_TARGET/1e6:.3f} MB/s")
    print(f"Chunk size: {CHUNK}")
    print("Running... (no printing in the loop)")

    t_start = time.perf_counter()
    t_end = t_start + DURATION_S

    frames = 0
    bad_loopback = 0
    bad_parse = 0
    crc_checked = 0
    crc_skipped = 0
    bad_crc = 0

    bytes_total = 0
    next_deadline = t_start

    try:
        seq = 0
        u0, s0 = get_cpu_time_seconds()
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

            # Framing: Magic, length, version, sanity
            ok, why = fast_validate_frame(rx)
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
    mb_s = bytes_total / elapsed / 1e6
    mbps = (bytes_total * 8) / elapsed / 1e6
    fps = frames / elapsed

    # Get CPU time and memory stats
    u1, s1 = get_cpu_time_seconds()
    proc_cpu = (u1 - u0) + (s1 - s0)
    cpu_pct_1core = 100.0 * proc_cpu / elapsed  # can exceed 100% if multithreaded (yours isn't)

    rss_mb, mem_total_mb, mem_avail_mb = get_mem_stats_linux()
    mem_used_pct = 100.0 * (mem_total_mb - mem_avail_mb) / mem_total_mb


    print_kv_table("RESULTS", [
    ("Elapsed", f"{elapsed:.3f} s"),
    ("Frames", f"{frames}"),
    ("FPS achieved", f"{fps:.2f}"),
    ("Data transferred", f"{bytes_total/1e6:.3f} MB"),
    ("Throughput", f"{mb_s:.3f} MB/s  ({mbps:.3f} Mb/s)"),
    ("Loopback mismatches", f"{bad_loopback}"),
    ("Frame validate fails", f"{bad_parse}"),
    ("CRC fails (1/N frames)", f"{bad_crc}  (every {CRC_EVERY_N})"),
    ("CRC checked", f"{crc_checked}"),
    ("CRC skipped (framing failed)", f"{crc_skipped}"),


    ])

    print_kv_table("RESOURCE USAGE", [
        ("Process CPU time", f"{proc_cpu:.3f} s"),
        ("Process CPU load", f"~{cpu_pct_1core:.1f}% of 1 core"),
        ("Process RSS", f"{rss_mb:.1f} MB"),
        ("System RAM used", f"{mem_used_pct:.1f}%"),
        ("System RAM avail", f"{mem_avail_mb:.0f} MB / {mem_total_mb:.0f} MB"),
        ("Mode", f"LIMITED {FPS_TARGET} FPS" if LIMIT_30_FPS_MODE else "UNTHROTTLED (max)"),
        ("Chunk size", f"{CHUNK} B"),
        ("Frame size", f"{FRAME_BYTES} B"),
        ("SCLK requested", f"{SPI_HZ/1e6:.1f} MHz"),
        ("SCLK reported", f"{sclk_reported_hz/1e6:.1f} MHz"),


    ])

if __name__ == "__main__":
    main()
