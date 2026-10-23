"""
Building blocks shared by the bitstream container and the entropy coders: bit-packed header fields, Morton order of the Gaussian positions,
lossless channel transforms, zigzag mapping of signed integers, and loading the C++ modules of compress/csrc.
"""
import hashlib
import importlib
import os
import struct

import numpy as np


def load_extension(module_name, source_name):
    """
    The C++ module module_name, after checking that it was built from the current compress/csrc/<source_name> (setup.py stores its hash).
    """
    source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", source_name)
    build = "pip install compress/csrc --no-build-isolation (from the repository root)"
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(f"The C++ module {module_name} is not installed: {build}") from error
    with open(source, "rb") as file:
        if module.SOURCE_HASH != hashlib.sha256(file.read()).hexdigest()[:16]:
            raise ImportError(f"The installed C++ module {module_name} was built from another version of {source}: {build}")
    return module


class BitWriter:
    def __init__(self):
        self.value, self.num_bits = 0, 0

    def write(self, value, num_bits):
        if num_bits > 0:
            if value < 0 or value >> num_bits:
                raise ValueError(f"Value {value} does not fit in {num_bits} bits")
            self.value = (self.value << num_bits) | value
            self.num_bits += num_bits

    def write_ue(self, value):
        """
        Exp-Golomb code (order 0) of a non-negative integer.
        """
        self.write(value + 1, 2 * (value + 1).bit_length() - 1)

    def write_se(self, value):
        self.write_ue(2 * value if value >= 0 else -2 * value - 1)

    def write_f32(self, value):
        self.write(struct.unpack("<I", struct.pack("<f", value))[0], 32)

    def to_bytes(self):
        padding = -self.num_bits % 8
        return ((self.value << padding).to_bytes((self.num_bits + padding) // 8, "big"))


class BitReader:
    def __init__(self, data, pos=0):
        self.data, self.bit_pos = data, 8 * pos

    def read(self, num_bits):
        value = 0
        for _ in range(num_bits):
            value = (value << 1) | ((self.data[self.bit_pos >> 3] >> (7 - (self.bit_pos & 7))) & 1)
            self.bit_pos += 1
        return value

    def read_ue(self, max_zeros=64):
        num_zeros = 0
        while self.read(1) == 0:
            num_zeros += 1
            if num_zeros > max_zeros:
                raise ValueError("Invalid Exp-Golomb code")
        return ((1 << num_zeros) | self.read(num_zeros)) - 1

    def read_se(self):
        value = self.read_ue()
        return value >> 1 if value % 2 == 0 else -(value >> 1) - 1

    def read_f32(self):
        return struct.unpack("<f", struct.pack("<I", self.read(32)))[0]

    def byte_pos(self):
        return (self.bit_pos + 7) // 8


def _part1by1(x):
    """
    Spread the lower 32 bits of x [uint64] to the even bit positions.
    """
    x = x & np.uint64(0x00000000FFFFFFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
    x = (x | (x << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x | (x << np.uint64(2))) & np.uint64(0x3333333333333333)
    x = (x | (x << np.uint64(1))) & np.uint64(0x5555555555555555)
    return x


def _compact1by1(x):
    """
    Inverse of _part1by1.
    """
    x = x & np.uint64(0x5555555555555555)
    x = (x | (x >> np.uint64(1))) & np.uint64(0x3333333333333333)
    x = (x | (x >> np.uint64(2))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x | (x >> np.uint64(4))) & np.uint64(0x00FF00FF00FF00FF)
    x = (x | (x >> np.uint64(8))) & np.uint64(0x0000FFFF0000FFFF)
    x = (x | (x >> np.uint64(16))) & np.uint64(0x00000000FFFFFFFF)
    return x


def morton_encode(x, y):
    return _part1by1(x.astype(np.uint64)) | (_part1by1(y.astype(np.uint64)) << np.uint64(1))


def morton_decode(codes):
    return _compact1by1(codes), _compact1by1(codes >> np.uint64(1))


def morton_order(positions):
    """
    Offsets of the position indices [P, 2], order of the Gaussians along the Morton curve, and the sorted Morton codes.
    """
    offsets = positions.min(axis=0)
    values = positions - offsets
    if values.max() >= 2**32:
        raise ValueError("The position must have at most 2^32 quantization levels per coordinate")
    codes = morton_encode(values[:, 0], values[:, 1])
    order = np.argsort(codes, kind="stable")
    return offsets, order, codes[order]


def zigzag(values):
    """
    Map signed integers [int64] to non-negative ones: 0, -1, 1, -2, ... -> 0, 1, 2, 3, ...
    """
    values = np.asarray(values, dtype=np.int64)
    return (values << 1) ^ (values >> 63)


def unzigzag(values):
    values = np.asarray(values, dtype=np.int64)
    return (values >> 1) ^ -(values & 1)


def check_channel_refs(refs, num_channels):
    """
    Reference channels must be valid channels that are not transformed themselves, so that the transform is inverted in a single pass.
    """
    if len(refs) != num_channels or any(not -1 <= ref < num_channels or ref == c or (ref >= 0 and refs[ref] >= 0) for c, ref in enumerate(refs)):
        raise ValueError(f"Invalid reference channels {list(refs)} for {num_channels} channels")


def subtract_channels(values, refs):
    """
    Lossless channel transform of integer values [P, C]: channel c is replaced by its difference with channel refs[c] (-1 keeps it),
    e.g., refs [1, -1, 1] codes R-G, G, B-G (the subtract-green transform of WebP lossless).
    """
    check_channel_refs(refs, values.shape[1])
    transformed = values.copy()
    for c, ref in enumerate(refs):
        if ref >= 0:
            transformed[:, c] -= values[:, ref]
    return transformed


def add_channels(values, refs):
    """
    Inverse of subtract_channels.
    """
    check_channel_refs(refs, values.shape[1])
    restored = values.copy()
    for c, ref in enumerate(refs):
        if ref >= 0:
            restored[:, c] += values[:, ref]
    return restored
