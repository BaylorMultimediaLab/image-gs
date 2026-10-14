"""
rANS entropy coding of integer streams with static token histograms.

Each value is split into a token and raw low bits (hybrid integer coding, as in JPEG XL): values below 2^split_exp are their own token,
larger values are coded by their bit length and their msb_bits most significant bits below the leading one, followed by the remaining
bits, which are coded as uniform. The token histogram of each stream is stored in the header with log-quantized counts, and all
probability tables are integers, so that decoding is deterministic.

The model search, rANS coding, and decoding are implemented in C++ (compress/csrc/rans.cpp, built as the module igs_rans); this module
writes and reads the stream headers.
"""
import hashlib
import os

import numpy as np

from compress.common import unzigzag, zigzag


def _load_extension():
    """
    The C++ coder, after checking that it was built from the current compress/csrc/rans.cpp.
    """
    source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "rans.cpp")
    build = "pip install compress/csrc --no-build-isolation (from the repository root)"
    try:
        import igs_rans
    except ImportError as error:
        raise ImportError(f"The C++ rANS coder is not installed: {build}") from error
    with open(source, "rb") as file:
        if igs_rans.SOURCE_HASH != hashlib.sha256(file.read()).hexdigest()[:16]:
            raise ImportError(f"The installed C++ rANS coder was built from another version of {source}: {build}")
    return igs_rans


_rans = _load_extension()


def ue_bits(values):
    """
    Lengths of the order-0 Exp-Golomb codes of non-negative integers [int64].
    """
    return 2 * bit_length(np.asarray(values, dtype=np.int64) + 1) - 1


def bit_length(values):
    """
    Bit lengths of non-negative integers [int64] below 2^53.
    """
    return np.frexp(np.asarray(values, dtype=np.float64))[1].astype(np.int64)


def table_bits(codes):
    """
    Size in bits of a table of count codes (after its length and mantissa_bits) and whether delta coding is cheaper than fixed width.
    """
    fixed = 5 + len(codes) * int(bit_length(codes.max()))
    delta = int(ue_bits(zigzag(np.diff(codes, prepend=0))).sum())
    return (1 + min(fixed, delta)), delta < fixed


def write_table(writer, codes, mantissa_bits):
    writer.write_ue(len(codes) - 1)
    writer.write(mantissa_bits, 2)
    _, use_delta = table_bits(codes)
    writer.write(int(use_delta), 1)
    if use_delta:
        for delta in np.diff(codes, prepend=0):
            writer.write_se(int(delta))
    else:
        width = int(bit_length(codes.max()))
        writer.write(width, 5)
        for code in codes:
            writer.write(int(code), width)


def read_table(reader):
    """
    Count codes of a token model and their mantissa_bits.
    """
    num_tokens = reader.read_ue() + 1
    mantissa_bits = reader.read(2)
    if reader.read(1):
        codes = np.cumsum([reader.read_se() for _ in range(num_tokens)])
    else:
        width = reader.read(5)
        codes = np.array([reader.read(width) for _ in range(num_tokens)], dtype=np.int64)
    return codes, mantissa_bits


def estimate_stream_bits(values):
    """
    Fast estimate of the coded size in bits of a stream of integers [int64], for rate decisions during optimization: the cheaper of the raw
    values and their differences, with ideal code lengths of their tokens (without most significant bits), the raw bits, and an approximate
    header cost per used token. Within about 3% of the size with fitted models (480p models, 450 to 3600 Gaussians) at a fraction of the time.
    """
    return _rans.estimate_stream_bits(values)


def encode_streams(groups, writer):
    """
    Encode groups (attributes) of integer streams [P] int64. Each stream is coded as raw values or as differences between consecutive
    Gaussians, whichever is smaller, mapped to non-negative integers by a shift or zigzag around its median, with the token model of the
    smallest total size. All streams share a single rANS state, and their models (token histograms) are written to writer.
    Returns the payload and the estimated size in bits of each group (models and payload).
    """
    payload, models = _rans.encode_streams([stream for streams in groups for stream in streams])
    models, group_bits = iter(models), []
    for streams in groups:
        num_bits, payload_bits = writer.num_bits, 0.0
        for _ in streams:
            delta, zigzag_mapped, param, split_exp, msb_bits, mantissa_bits, codes, stream_payload_bits = next(models)
            writer.write(delta, 1)
            writer.write(zigzag_mapped, 1)
            writer.write_se(param)
            writer.write(split_exp, 4)
            writer.write(msb_bits, 2)
            write_table(writer, codes, mantissa_bits)
            payload_bits += stream_payload_bits
        group_bits.append(writer.num_bits - num_bits + payload_bits)
    return payload, group_bits


def _read_stream(reader):
    delta, zigzag_mapped, param = reader.read(1), reader.read(1), reader.read_se()
    split_exp, msb_bits = reader.read(4), reader.read(2)
    codes, mantissa_bits = read_table(reader)
    return delta, zigzag_mapped, param, (split_exp, msb_bits, mantissa_bits, codes)


def decode_streams(reader, data, num_streams, count):
    """
    Decode groups of num_streams[i] streams of count values, with the models read from reader and the payload following the header in data.
    """
    headers = [[_read_stream(reader) for _ in range(num)] for num in num_streams]
    mapped = iter(_rans.decode_streams(data[reader.byte_pos():], [model for streams in headers for *_, model in streams], count))
    groups = []
    for streams in headers:
        groups.append([])
        for delta, zigzag_mapped, param, _ in streams:
            values = next(mapped)
            values = (unzigzag(values) if zigzag_mapped else values) + param
            groups[-1].append(np.cumsum(values) if delta else values)
    return groups
