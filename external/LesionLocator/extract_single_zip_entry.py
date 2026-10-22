#!/usr/bin/env python3
"""Extract one ranged ZIP entry without downloading the complete archive.

The first input chunk must begin at the entry's local-file header. Additional
chunks are treated as a byte-for-byte continuation. This is intentionally
small and only supports stored/deflated, unencrypted entries.
"""

from __future__ import annotations

import argparse
import binascii
import os
import struct
import zlib
from pathlib import Path
from typing import BinaryIO, Iterable


LOCAL_HEADER = struct.Struct("<4s5H3I2H")


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    value = stream.read(count)
    if len(value) != count:
        raise EOFError(f"needed {count} bytes, received {len(value)}")
    return value


def extract(
    chunks: Iterable[Path],
    output: Path,
    compressed_size: int,
    uncompressed_size: int,
    expected_crc32: int,
) -> None:
    chunks = list(chunks)
    if not chunks:
        raise ValueError("at least one input chunk is required")

    streams = [path.open("rb") for path in chunks]
    temporary = output.with_name(output.name + ".part")
    try:
        header = LOCAL_HEADER.unpack(_read_exact(streams[0], LOCAL_HEADER.size))
        signature, _, flags, method, _, _, _, _, _, name_len, extra_len = header
        if signature != b"PK\x03\x04":
            raise ValueError("first chunk does not start with a ZIP local header")
        if flags & 0x1:
            raise ValueError("encrypted ZIP entries are unsupported")
        if method not in (0, 8):
            raise ValueError(f"unsupported ZIP compression method: {method}")

        filename = _read_exact(streams[0], name_len).decode("utf-8")
        _read_exact(streams[0], extra_len)
        output.parent.mkdir(parents=True, exist_ok=True)

        decompressor = zlib.decompressobj(-15) if method == 8 else None
        remaining = compressed_size
        written = 0
        crc32 = 0
        with temporary.open("wb") as destination:
            for stream in streams:
                while remaining:
                    block = stream.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        break
                    remaining -= len(block)
                    decoded = decompressor.decompress(block) if decompressor else block
                    if decoded:
                        destination.write(decoded)
                        written += len(decoded)
                        crc32 = binascii.crc32(decoded, crc32)
                if remaining == 0:
                    break
            if decompressor:
                decoded = decompressor.flush()
                destination.write(decoded)
                written += len(decoded)
                crc32 = binascii.crc32(decoded, crc32)

        if remaining:
            raise EOFError(f"entry is missing {remaining} compressed bytes")
        if written != uncompressed_size:
            raise ValueError(f"size mismatch: expected {uncompressed_size}, wrote {written}")
        if crc32 & 0xFFFFFFFF != expected_crc32:
            raise ValueError(
                f"CRC mismatch: expected {expected_crc32:08x}, got {crc32 & 0xFFFFFFFF:08x}"
            )
        os.replace(temporary, output)
        print(f"extracted {filename} -> {output} ({written} bytes, CRC verified)")
    finally:
        for stream in streams:
            stream.close()
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compressed-size", type=int, required=True)
    parser.add_argument("--uncompressed-size", type=int, required=True)
    parser.add_argument("--crc32", required=True, help="expected hexadecimal CRC32")
    args = parser.parse_args()
    extract(
        args.input,
        args.output,
        args.compressed_size,
        args.uncompressed_size,
        int(args.crc32, 16),
    )


if __name__ == "__main__":
    main()
