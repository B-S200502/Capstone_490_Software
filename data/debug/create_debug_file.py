#!/usr/bin/env python3
"""
Create a 1.1 GB dummy file in data/debug/ for testing (e.g. storage bar, delete).

Run from repo root: python3 data/debug/create_debug_file.py
"""
from pathlib import Path

DIR = Path(__file__).resolve().parent
SIZE_MB = 1126  # ~1.1 GB
CHUNK_MB = 64
FILENAME = "dummy_1.1gb.bin"


def main() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    path = DIR / FILENAME
    if path.exists():
        print(f"File already exists: {path}")
        print("Delete it first if you want to regenerate.")
        return
    print(f"Creating {SIZE_MB} MB file at {path} ...")
    written = 0
    chunk = b"\0" * (CHUNK_MB * 1024 * 1024)
    with open(path, "wb") as f:
        for _ in range(SIZE_MB // CHUNK_MB):
            f.write(chunk)
            written += CHUNK_MB
            print(f"  {written} MB ...", end="\r")
        remainder = SIZE_MB % CHUNK_MB
        if remainder:
            f.write(b"\0" * (remainder * 1024 * 1024))
            written += remainder
    print(f"\nDone. Created {path} ({written} MB).")


if __name__ == "__main__":
    main()
