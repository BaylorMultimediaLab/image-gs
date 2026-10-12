import lzma
import struct

import numpy as np

MAGIC = b"IGS0"
LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
DELTA = 1  # Attribute flag: values are coded as differences between consecutive Gaussians


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


def _zigzag(x):
    """
    Map signed integers [int64] to non-negative ones: 0, -1, 1, -2, ... -> 0, 1, 2, 3, ...
    """
    return ((x << 1) ^ (x >> 63)).astype(np.uint64)


def _unzigzag(z):
    z = z.astype(np.int64)
    return (z >> 1) ^ -(z & 1)


def _compress(values):
    """
    LZMA-compress non-negative integers [uint64] stored as separate byte planes, keeping only the planes needed for the largest value.
    """
    width = max(1, (int(values.max()).bit_length() + 7) // 8)
    planes = values.astype("<u8").view(np.uint8).reshape(-1, 8)[:, :width].T
    return width, lzma.compress(planes.tobytes(), format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)


def _decompress(payload, width, count):
    planes = np.frombuffer(lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS), dtype=np.uint8).reshape(width, count)
    values = np.zeros((count, 8), dtype=np.uint8)
    values[:, :width] = planes.T
    return values.view("<u8").reshape(-1)


def encode_bitstream(attributes, meta=b""):
    """
    Encode quantized Gaussian attributes into a bitstream.
    attributes: list of (indices [P, C] int64, steps [C] float64, origins [C] float64), where the first attribute is the position (C = 2).
    Gaussians are sorted along the Morton curve of their positions and the sorted Morton codes are delta-coded. The other attributes
    follow in the same order, each stored as raw values or as differences between consecutive Gaussians, whichever is smaller.
    Returns the bitstream, the order of the Gaussians in it, and the number of bytes of each attribute.
    """
    num_gaussians = len(attributes[0][0])
    data = [MAGIC, struct.pack("<IBB", num_gaussians, len(meta), len(attributes)), meta]
    order = None
    attribute_bytes = []
    for idx, (indices, steps, origins) in enumerate(attributes):
        num_channels = indices.shape[1]
        offsets = indices.min(axis=0)
        values = indices - offsets
        if idx == 0:
            if num_channels != 2 or values.max() >= 2**32:
                raise ValueError("The first attribute must be the position with at most 2^32 quantization levels per coordinate")
            codes = morton_encode(values[:, 0], values[:, 1])
            order = np.argsort(codes, kind="stable")
            flags = DELTA
            width, payload = _compress(np.diff(codes[order], prepend=np.uint64(0)))
        else:
            values = values[order].T
            candidates = [(0, _compress(values.astype(np.uint64).reshape(-1))),
                          (DELTA, _compress(_zigzag(np.diff(values, axis=1, prepend=0)).reshape(-1)))]
            flags, (width, payload) = min(candidates, key=lambda candidate: len(candidate[1][1]))
        record = struct.pack("<B", num_channels)
        record += b"".join(struct.pack("<ddq", float(step), float(origin), int(offset)) for step, origin, offset in zip(steps, origins, offsets))
        record += struct.pack("<BBI", flags, width, len(payload)) + payload
        data.append(record)
        attribute_bytes.append(len(record))
    return b"".join(data), order, attribute_bytes


def decode_bitstream(data):
    """
    Decode a bitstream written by encode_bitstream.
    Returns the attributes (with the Gaussians in bitstream order) and the metadata.
    """
    if data[:len(MAGIC)] != MAGIC:
        raise ValueError("Not an Image-GS bitstream")
    pos = len(MAGIC)
    num_gaussians, meta_len, num_attributes = struct.unpack_from("<IBB", data, pos)
    pos += struct.calcsize("<IBB")
    meta = data[pos:pos+meta_len]
    pos += meta_len
    attributes = []
    for idx in range(num_attributes):
        num_channels, = struct.unpack_from("<B", data, pos)
        pos += 1
        channel_params = [struct.unpack_from("<ddq", data, pos + c * struct.calcsize("<ddq")) for c in range(num_channels)]
        pos += num_channels * struct.calcsize("<ddq")
        steps = np.array([params[0] for params in channel_params], dtype=np.float64)
        origins = np.array([params[1] for params in channel_params], dtype=np.float64)
        offsets = np.array([params[2] for params in channel_params], dtype=np.int64)
        flags, width, payload_len = struct.unpack_from("<BBI", data, pos)
        pos += struct.calcsize("<BBI")
        payload = data[pos:pos+payload_len]
        pos += payload_len
        if idx == 0:
            codes = np.cumsum(_decompress(payload, width, num_gaussians), dtype=np.uint64)
            values = np.stack(morton_decode(codes), axis=1).astype(np.int64)
        else:
            values = _decompress(payload, width, num_channels * num_gaussians).reshape(num_channels, num_gaussians)
            values = np.cumsum(_unzigzag(values), axis=1) if flags & DELTA else values.astype(np.int64)
            values = values.T
        attributes.append((values + offsets, steps, origins))
    return attributes, meta
