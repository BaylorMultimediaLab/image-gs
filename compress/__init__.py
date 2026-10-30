"""
Bitstream container of quantized Gaussian attributes: Gaussians are sorted along the Morton curve of their positions, positions are coded
as gaps between sorted Morton codes, optional lossless channel transforms are applied (e.g., subtract-green), and the quantization
parameters are stored in a bit-packed header. The resulting integer streams are entropy-coded with rANS (compress.rans_codec).

Two-stage positions (refine_bits > 0): the Morton curve and the gaps are those of coarse positions (the position indices shifted right by
refine_bits), and the low bits of each coordinate follow as raw refinement bits, of which a Gaussian may omit the lowest e (per-axis
exponents [P, 2], its position indices being multiples of 2^e). The exponents may depend on the other attributes: a decoder decodes all
streams, reconstructs the other attributes and the exponents (decode_bitstream's reconstruct), and then reads the refinement.
"""
import struct

import numpy as np

import compress.rans_codec as rans_codec
from compress.common import BitReader, BitWriter, add_channels, morton_decode, morton_encode, morton_order, subtract_channels

MAGIC = b"IGS1"
HEADER = "<4sBIBB"  # Magic, codec (low 4 bits) and refinement bits (high 4 bits), number of Gaussians, metadata length, number of attributes
CODECS = {"rans": (1, rans_codec)}  # Entropy coders by name: id in the header, module


def _write_params(writer, steps, origins):
    """
    Dequantization steps and origins [C] as single-precision values, written once when shared by all channels.
    """
    if np.any(np.float32(steps) != steps) or np.any(np.float32(origins) != origins):
        raise ValueError("Quantization steps and origins must be single-precision values")
    shared = bool(np.all(steps == steps[0]) and np.all(origins == origins[0]))
    writer.write(int(shared), 1)
    for step, origin in zip(steps[:1] if shared else steps, origins[:1] if shared else origins):
        writer.write_f32(float(step))
        writer.write_f32(float(origin))


def _read_params(reader, num_channels):
    shared = reader.read(1)
    params = [(reader.read_f32(), reader.read_f32()) for _ in range(1 if shared else num_channels)]
    params = params * num_channels if shared else params
    return np.array([step for step, _ in params], dtype=np.float64), np.array([origin for _, origin in params], dtype=np.float64)


def bitstream_order(positions, refine_bits=0):
    """
    Order of Gaussians with the position indices [P, 2] in the bitstream: along the Morton curve of their coarse positions (positions >>
    refine_bits), within a coarse cell by the Morton code of their positions. Returns the offsets of the coarse positions, the order, and the
    sorted Morton codes of the coarse positions.
    """
    if refine_bits == 0:
        return morton_order(positions)
    coarse = positions >> refine_bits
    offsets = coarse.min(axis=0)
    if (coarse - offsets).max() >= 2**32:
        raise ValueError("The position must have at most 2^32 quantization levels per coordinate")
    codes = morton_encode(*(coarse - offsets).T)
    fine = morton_encode(*(positions - positions.min(axis=0)).T)
    order = np.lexsort((fine, codes))
    return offsets, order, codes[order]


def _refinement_fields(positions, exponents, refine_bits):
    """
    Raw refinement fields of two-stage positions in bitstream order: values and bit lengths [2 P] (x, then y, of each Gaussian).
    """
    low = positions & ((1 << refine_bits) - 1)
    if np.any(low & ((1 << exponents) - 1)):
        raise ValueError("Position indices must be multiples of 2^exponent")
    return (low >> exponents).reshape(-1), (refine_bits - exponents).reshape(-1)


def _pack_bits(values, lengths):
    """
    Concatenated big-endian bit fields as an integer, and its number of bits.
    """
    total = int(lengths.sum())
    if total == 0:
        return 0, 0
    starts = np.cumsum(lengths) - lengths
    bits = np.zeros(total, dtype=np.uint8)
    for j in range(int(lengths.max())):
        has = lengths > j
        bits[starts[has] + j] = (values[has] >> (lengths[has] - 1 - j)) & 1
    packed = np.packbits(bits)
    return int.from_bytes(packed.tobytes(), "big") >> (8 * len(packed) - total), total


def _unpack_bits(data, bit_pos, lengths):
    """
    Bit fields of the given lengths that start at bit_pos of data, as written by _pack_bits.
    """
    total = int(lengths.sum())
    first = bit_pos // 8
    bits = np.unpackbits(np.frombuffer(data[first:(bit_pos + total + 7) // 8], dtype=np.uint8))[bit_pos - 8 * first:][:total]
    if len(bits) != total:
        raise ValueError("Truncated position refinement")
    starts = np.cumsum(lengths) - lengths
    values = np.zeros(len(lengths), dtype=np.int64)
    for j in range(int(lengths.max(initial=0))):
        has = lengths > j
        values[has] = (values[has] << 1) | bits[starts[has] + j]
    return values


def encode_bitstream(attributes, meta=b"", codec="rans", channel_refs=None, refine_bits=0, refine_exponents=None):
    """
    Encode quantized Gaussian attributes into a bitstream with the given entropy coder (CODECS).
    attributes: list of (indices [P, C] int64, steps [C] float64, origins [C] float64), where the first attribute is the position (C = 2).
    channel_refs: optional reference channels per attribute (None or a list [C], see compress.common.subtract_channels), e.g.,
    [None, None, None, [1, -1, 1]] codes the features as R-G, G, B-G.
    refine_bits: two-stage positions with this many refinement bits per coordinate (0: one stage), of which each Gaussian omits the lowest
    refine_exponents [P, 2] (int64, in the order of the attributes; zeros if not given).
    Returns the bitstream, the order of the Gaussians in it (bitstream_order), and the number of bytes of each attribute (estimated for rANS,
    whose attributes share a single stream).
    """
    if codec not in CODECS:
        raise ValueError(f"Unsupported bitstream codec '{codec}', valid values include {', '.join(repr(name) for name in CODECS)}")
    if not 0 <= refine_bits < 16:
        raise ValueError(f"Refinement bits must be between 0 and 15 but got: {refine_bits}")
    channel_refs = [None] * len(attributes) if channel_refs is None else channel_refs
    if len(channel_refs) != len(attributes) or channel_refs[0] is not None:
        raise ValueError("Channel references must be given per attribute, and the position cannot be transformed")
    writer, groups, header_bits, order = BitWriter(), [], [], None
    for idx, ((indices, steps, origins), refs) in enumerate(zip(attributes, channel_refs)):
        num_bits = writer.num_bits
        num_channels = indices.shape[1]
        writer.write_ue(num_channels - 1)
        _write_params(writer, steps, origins)
        if idx == 0:
            if num_channels != 2:
                raise ValueError("The first attribute must be the position")
            offsets, order, codes = bitstream_order(indices, refine_bits)
            for offset in offsets:
                writer.write_se(int(offset))
            groups.append([np.diff(codes, prepend=np.uint64(0)).astype(np.int64)])
        else:
            refs = [-1] * num_channels if refs is None else list(refs)
            for ref in refs:
                writer.write_ue(ref + 1)
            values = subtract_channels(indices[order], refs)
            groups.append([values[:, c] for c in range(num_channels)])
        header_bits.append(writer.num_bits - num_bits)
    if refine_bits > 0:
        exponents = np.zeros_like(attributes[0][0]) if refine_exponents is None else refine_exponents
        if np.any(exponents < 0) or np.any(exponents > refine_bits):
            raise ValueError(f"Refinement exponents must be between 0 and {refine_bits}")
        num_bits = writer.num_bits
        value, total = _pack_bits(*_refinement_fields(attributes[0][0][order], exponents[order], refine_bits))
        writer.write_ue(total)
        writer.write(value, total)
        header_bits[0] += writer.num_bits - num_bits
    codec_id, module = CODECS[codec]
    payload, group_bits = module.encode_streams(groups, writer)
    header = struct.pack(HEADER, MAGIC, codec_id | refine_bits << 4, len(order), len(meta), len(attributes)) + meta
    data = header + writer.to_bytes() + payload
    return data, order, [(bits + coded_bits) / 8.0 for bits, coded_bits in zip(header_bits, group_bits)]


def estimate_bits(indices, channel_refs=None):
    """
    Fast estimate of the coded size in bits of quantized Gaussian attributes without the header, with the transforms of encode_bitstream
    and rans_codec.estimate_stream_bits.
    indices: list of quantization indices [P, C] int64 per attribute, where the first attribute is the position (C = 2).
    """
    channel_refs = [None] * len(indices) if channel_refs is None else channel_refs
    _, order, codes = morton_order(indices[0])
    streams = [np.diff(codes, prepend=np.uint64(0)).astype(np.int64)]
    for values, refs in zip(indices[1:], channel_refs[1:]):
        values = values[order] if refs is None else subtract_channels(values[order], refs)
        streams += [values[:, c] for c in range(values.shape[1])]
    return sum(rans_codec.estimate_stream_bits(stream) for stream in streams)


def decode_bitstream(data, reconstruct=None):
    """
    Decode a bitstream written by encode_bitstream.
    reconstruct(attributes) -> (attributes, refine_exponents): optional transform of the attributes as coded (with the positions of two-stage
    positions as coarse positions << refine_bits) into their indices, which also gives the refinement exponents [P, 2] of two-stage positions
    in bitstream order (None for zeros); bitstreams with two-stage positions require it.
    Returns the attributes (with the Gaussians in bitstream order) and the metadata.
    """
    magic, codec_byte, num_gaussians, meta_len, num_attributes = struct.unpack_from(HEADER, data)
    codec_id, refine_bits = codec_byte & 15, codec_byte >> 4
    modules = dict(CODECS.values())
    if magic != MAGIC or codec_id not in modules:
        raise ValueError("Not an Image-GS bitstream")
    pos = struct.calcsize(HEADER)
    meta = data[pos:pos+meta_len]
    reader = BitReader(data, pos + meta_len)
    params = []
    for idx in range(num_attributes):
        num_channels = reader.read_ue() + 1
        steps, origins = _read_params(reader, num_channels)
        if idx == 0:
            extra = np.array([reader.read_se() for _ in range(num_channels)], dtype=np.int64)  # Position offsets
        else:
            extra = [reader.read_ue() - 1 for _ in range(num_channels)]  # Reference channels
        params.append((num_channels, steps, origins, extra))
    if refine_bits > 0:
        refine_total = reader.read_ue()
        refine_pos = reader.bit_pos
        reader.bit_pos += refine_total
    num_streams = [1] + [num_channels for num_channels, _, _, _ in params[1:]]
    groups = modules[codec_id].decode_streams(reader, data, num_streams, num_gaussians)
    attributes = []
    for idx, ((_, steps, origins, extra), streams) in enumerate(zip(params, groups)):
        if idx == 0:
            codes = np.cumsum(streams[0].astype(np.uint64), dtype=np.uint64)
            values = (np.stack(morton_decode(codes), axis=1).astype(np.int64) + extra) << refine_bits
        else:
            values = add_channels(np.stack(streams, axis=1), extra)
        attributes.append((values, steps, origins))
    exponents = None
    if reconstruct is not None:
        attributes, exponents = reconstruct(attributes)
    if refine_bits > 0:
        if reconstruct is None:
            raise ValueError("Decoding two-stage positions requires reconstruct (the refinement exponents)")
        exponents = np.zeros((num_gaussians, 2), dtype=np.int64) if exponents is None else np.asarray(exponents, dtype=np.int64)
        lengths = (refine_bits - exponents).reshape(-1)
        if lengths.sum() != refine_total:
            raise ValueError("The refinement exponents do not match the bitstream")
        low = _unpack_bits(data, refine_pos, lengths).reshape(-1, 2) << exponents
        positions, steps, origins = attributes[0]
        attributes = [(positions + low, steps, origins)] + list(attributes[1:])
    return attributes, meta
