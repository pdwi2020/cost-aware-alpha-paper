#!/usr/bin/env python3
"""Set SG_READ_ONLY (0x10) on the __DATA_CONST LC_SEGMENT_64 of a thin arm64
Mach-O so macOS 26 dyld will load it. Lean v4.16.0's linker emits flags=0x0,
which newer dyld rejects. Surgical: only the one uint32 flags field is touched."""
import struct, sys, shutil, os

MH_MAGIC_64 = 0xFEEDFACF
LC_SEGMENT_64 = 0x19
SG_READ_ONLY = 0x10

path = sys.argv[1]
shutil.copy2(path, path + ".prepatch.bak")
with open(path, "rb") as f:
    data = bytearray(f.read())

magic, = struct.unpack_from("<I", data, 0)
assert magic == MH_MAGIC_64, f"not a thin arm64 Mach-O (magic={magic:#x})"
ncmds, = struct.unpack_from("<I", data, 16)

off = 32  # past mach_header_64
patched = 0
for _ in range(ncmds):
    cmd, cmdsize = struct.unpack_from("<II", data, off)
    if cmd == LC_SEGMENT_64:
        segname = bytes(data[off + 8:off + 24]).rstrip(b"\x00")
        if segname == b"__DATA_CONST":
            flags_off = off + 68
            cur, = struct.unpack_from("<I", data, flags_off)
            new = cur | SG_READ_ONLY
            struct.pack_into("<I", data, flags_off, new)
            print(f"__DATA_CONST flags {cur:#x} -> {new:#x} at file offset {flags_off}")
            patched += 1
    off += cmdsize

assert patched == 1, f"expected exactly one __DATA_CONST segment, patched {patched}"
with open(path, "wb") as f:
    f.write(data)
print("patched OK ->", path)
