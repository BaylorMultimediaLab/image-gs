"""
LZMA entropy coding of integer streams, stored as byte planes.
"""
import lzma

import numpy as np

from compress.common import unzigzag, zigzag

LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
DELTA = 1  # Group flag: values are coded as differences between consecutive Gaussians


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


def encode_streams(groups, writer):
    """
    Encode groups (attributes) of integer streams [P] int64. The streams of a group are compressed together, as raw values (minus their
    minimum) or as differences between consecutive Gaussians, whichever is smaller. The side information is written to writer.
    Returns the payload and the size in bits of each group (side information and compressed data).
    """
    payloads, group_bits = [], []
    for streams in groups:
        num_bits = writer.num_bits
        values = np.stack(streams)
        offsets = values.min(axis=1, keepdims=True)
        values = values - offsets
        candidates = [(0, _compress(values.astype(np.uint64).reshape(-1))),
                      (DELTA, _compress(zigzag(np.diff(values, axis=1, prepend=0)).astype(np.uint64).reshape(-1)))]
        flags, (width, payload) = min(candidates, key=lambda candidate: len(candidate[1][1]))
        for offset in offsets[:, 0]:
            writer.write_se(int(offset))
        writer.write(flags, 1)
        writer.write(width - 1, 3)
        writer.write_ue(len(payload))
        payloads.append(payload)
        group_bits.append(writer.num_bits - num_bits + 8 * len(payload))
    return b"".join(payloads), group_bits


def decode_streams(reader, data, num_streams, count):
    """
    Decode groups of num_streams[i] streams of count values, with the side information read from reader and the payload following the header in data.
    """
    headers = []
    for num in num_streams:
        offsets = np.array([reader.read_se() for _ in range(num)], dtype=np.int64)
        flags, width, length = reader.read(1), reader.read(3) + 1, reader.read_ue()
        headers.append((offsets, flags, width, length))
    pos = reader.byte_pos()
    groups = []
    for num, (offsets, flags, width, length) in zip(num_streams, headers):
        values = _decompress(data[pos:pos+length], width, num * count).reshape(num, count)
        pos += length
        values = np.cumsum(unzigzag(values), axis=1) if flags & DELTA else values.astype(np.int64)
        groups.append(list(values + offsets[:, None]))
    return groups
