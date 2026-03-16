import re
import struct
import serial
import time

MAGIC = 0xAABBCCDD
HEADER_LEN = 28

HEX_RE = re.compile(r"\b[0-9A-Fa-f]{2}\b")

def bytes_from_hex_lines(lines):
    out = bytearray()
    for ln in lines:
        out.extend(int(x, 16) for x in HEX_RE.findall(ln))
    return bytes(out)

def parse_header(h: bytes):
    # SPI_FrameHeader_t packed, little-endian
    return struct.unpack("<IHHIHHIHHI", h)

def main():
    port = "/dev/ttyACM0"
    baud = 115200

    ser = serial.Serial(port, baudrate=baud, timeout=1)
    ser.reset_input_buffer()

    print(f"Listening (TEXT mode) on {port} @ {baud}")
    print("Waiting for 'Hex dump (first 64 bytes):' ...\n")

    collecting = False
    hex_lines = []

    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue

            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            if not line:
                continue

            # Uncomment if you want to see everything:
            # print(line)

            if "Hex dump (first 64 bytes)" in line:
                collecting = True
                hex_lines = []
                continue

            if collecting:
                # Expect 4 lines of 16 bytes each (64 bytes total)
                if HEX_RE.search(line):
                    hex_lines.append(line)
                # Once we have 4 lines, parse
                if len(hex_lines) >= 4:
                    collecting = False
                    blob = bytes_from_hex_lines(hex_lines[:4])

                    if len(blob) < 64:
                        print(f"[warn] got only {len(blob)} bytes, skipping")
                        continue

                    # First 28 bytes are header, next payload starts
                    header = blob[:HEADER_LEN]
                    (magic, version, header_len, frame_counter, mic_count, fft_size,
                     sample_rate, bin_count, reserved_adc, payload_len) = parse_header(header)

                    if magic != MAGIC or header_len != HEADER_LEN:
                        print("[warn] bad magic/header_len, resyncing...")
                        continue

                    # The hex dump only shows first 64 bytes total, so payload might be partial.
                    payload_available = blob[HEADER_LEN:64]
                    first_bins = ""
                    if len(payload_available) >= 16:
                        re0, im0 = struct.unpack("<ff", payload_available[0:8])
                        re1, im1 = struct.unpack("<ff", payload_available[8:16])
                        first_bins = f" bins0=({re0:+.3f},{im0:+.3f}) bins1=({re1:+.3f},{im1:+.3f})"

                    print(
                        f"frame={frame_counter} adc(res)={reserved_adc} "
                        f"mics={mic_count} fft={fft_size} sr={sample_rate} bins={bin_count} "
                        f"payload_len={payload_len}{first_bins}"
                    )

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
