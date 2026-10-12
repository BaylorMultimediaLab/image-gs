"""
Bitstream container of quantized Gaussian attributes. The steps are independent of the entropy coder: Gaussians are sorted along the
Morton curve of their positions, positions are coded as gaps between sorted Morton codes, optional lossless channel transforms are
applied (e.g., subtract-green), and the quantization parameters are stored in a bit-packed header. The resulting integer streams are
entropy-coded with LZMA (compress.lzma_codec) or rANS (compress.rans_codec).
"""
import struct

import numpy as np

import compress.lzma_codec as lzma_codec
import compress.rans_codec as rans_codec
from compress.common import BitReader, BitWriter, add_channels, morton_decode, morton_order, subtract_channels

MAGIC = b"IGS1"
HEADER = "<4sBIBB"  # Magic, codec, number of Gaussians, metadata length, number of attributes
CODECS = {"lzma": lzma_codec, "rans": rans_codec}  # Codec ids follow this order


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


def encode_bitstream(attributes, meta=b"", codec="lzma", channel_refs=None):
    """
    Encode quantized Gaussian attributes into a bitstream with the given entropy coder ("lzma" or "rans").
    attributes: list of (indices [P, C] int64, steps [C] float64, origins [C] float64), where the first attribute is the position (C = 2).
    channel_refs: optional reference channels per attribute (None or a list [C], see compress.common.subtract_channels), e.g.,
    [None, None, None, [1, -1, 1]] codes the features as R-G, G, B-G.
    Returns the bitstream, the order of the Gaussians in it, and the number of bytes of each attribute (estimated for rANS,
    whose attributes share a single stream).
    """
    if codec not in CODECS:
        raise ValueError(f"Unsupported bitstream codec '{codec}', valid values include {', '.join(repr(name) for name in CODECS)}")
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
            offsets, order, codes = morton_order(indices)
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
    payload, group_bits = CODECS[codec].encode_streams(groups, writer)
    header = struct.pack(HEADER, MAGIC, list(CODECS).index(codec), len(order), len(meta), len(attributes)) + meta
    data = header + writer.to_bytes() + payload
    return data, order, [(bits + coded_bits) / 8.0 for bits, coded_bits in zip(header_bits, group_bits)]


def decode_bitstream(data):
    """
    Decode a bitstream written by encode_bitstream.
    Returns the attributes (with the Gaussians in bitstream order) and the metadata.
    """
    magic, codec_id, num_gaussians, meta_len, num_attributes = struct.unpack_from(HEADER, data)
    if magic != MAGIC or codec_id >= len(CODECS):
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
    num_streams = [1] + [num_channels for num_channels, _, _, _ in params[1:]]
    groups = list(CODECS.values())[codec_id].decode_streams(reader, data, num_streams, num_gaussians)
    attributes = []
    for idx, ((_, steps, origins, extra), streams) in enumerate(zip(params, groups)):
        if idx == 0:
            codes = np.cumsum(streams[0].astype(np.uint64), dtype=np.uint64)
            values = np.stack(morton_decode(codes), axis=1).astype(np.int64) + extra
        else:
            values = add_channels(np.stack(streams, axis=1), extra)
        attributes.append((values, steps, origins))
    return attributes, meta
