"""
rANS entropy coding of integer streams with static token histograms.

Each value is split into a token and raw low bits (hybrid integer coding, as in JPEG XL): values below 2^split_exp are their own token,
larger values are coded by their bit length and their msb_bits most significant bits below the leading one, followed by the remaining
bits, which are coded as uniform. The token histogram of each stream is stored in the header with log-quantized counts, and all
probability tables are integers, so that decoding is deterministic.
"""
import numpy as np

from compress.common import unzigzag, zigzag

PREC = 16  # Precision of the token models (frequencies sum to 2^PREC)
RAW_CHUNK = 16  # Maximum number of raw bits per rANS symbol
RANS_L = 1 << 31  # Lower bound of the rANS state (64-bit state with 32-bit renormalization, as in ryg_rans' rans64)
MAX_SPLIT_EXP = 12
MAX_MSB_BITS = 3
MAX_MANTISSA_BITS = 3
TABLE_BITS_PER_TOKEN = 6  # Approximate header cost of each used token in estimate_stream_bits


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


def tokenize(values, split_exp, msb_bits):
    """
    Tokens, raw bits, and numbers of raw bits of non-negative integers [int64].
    """
    split = 1 << split_exp
    exponents = bit_length(values) - 1
    large = values >= split
    num_raw = np.where(large, exponents - msb_bits, 0)
    msbs = (values >> np.maximum(num_raw, 0)) & ((1 << msb_bits) - 1)
    tokens = np.where(large, split + ((exponents - split_exp) << msb_bits) + msbs, values)
    raw = values & ((1 << num_raw) - 1)
    return tokens, raw, num_raw


def detokenize(token, split_exp, msb_bits):
    """
    Value without its raw bits and number of raw bits of a single token.
    """
    split = 1 << split_exp
    if token < split:
        return token, 0
    exponent = split_exp + ((token - split) >> msb_bits)
    num_raw = exponent - msb_bits
    return (((1 << msb_bits) | ((token - split) & ((1 << msb_bits) - 1))) << num_raw), num_raw


def quantize_counts(counts, mantissa_bits):
    """
    Log-quantized codes of token counts: 0 for unused tokens, otherwise (exponent << mantissa_bits) + mantissa, with weight mantissa << exponent.
    """
    counts = np.asarray(counts, dtype=np.int64)
    small = counts < (2 << mantissa_bits)
    exponents = np.maximum(bit_length(counts) - (mantissa_bits + 1), 0)
    mantissas = (counts + np.where(exponents > 0, 1 << np.maximum(exponents - 1, 0), 0)) >> exponents
    overflow = mantissas == (2 << mantissa_bits)
    mantissas = np.where(overflow, 1 << mantissa_bits, mantissas)
    exponents = exponents + overflow
    return np.where(small, counts, (exponents << mantissa_bits) + mantissas)


def code_weights(codes, mantissa_bits):
    codes = np.asarray(codes, dtype=np.int64)
    small = codes < (2 << mantissa_bits)
    exponents = np.where(small, 0, (codes >> mantissa_bits) - 1)
    mantissas = np.where(small, codes, codes - (exponents << mantissa_bits))
    return mantissas << exponents


def normalize(weights):
    """
    Integer frequencies summing to 2^PREC, positive for the tokens with positive weight.
    """
    weights = np.asarray(weights, dtype=np.int64)
    freqs = (weights << PREC) // weights.sum()
    freqs = np.where(weights > 0, np.maximum(freqs, 1), 0)
    largest = int(np.argmax(freqs))
    freqs[largest] += (1 << PREC) - freqs.sum()
    if freqs[largest] < 1:
        raise ValueError("Cannot normalize the token histogram")
    return freqs


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
    num_tokens = reader.read_ue() + 1
    mantissa_bits = reader.read(2)
    if reader.read(1):
        codes = np.cumsum([reader.read_se() for _ in range(num_tokens)])
    else:
        width = reader.read(5)
        codes = np.array([reader.read(width) for _ in range(num_tokens)], dtype=np.int64)
    return normalize(code_weights(codes, mantissa_bits))


class StreamModel:
    """
    Token model of a stream of non-negative integers, chosen to minimize the total size (header and payload).
    """

    def __init__(self, split_exp, msb_bits, mantissa_bits, codes, payload_bits, bits):
        self.split_exp, self.msb_bits, self.mantissa_bits = split_exp, msb_bits, mantissa_bits
        self.codes, self.payload_bits, self.bits = codes, payload_bits, bits  # Estimated payload size and total size with the model
        self.freqs = normalize(code_weights(codes, mantissa_bits))

    @classmethod
    def fit(cls, values):
        best = None
        max_exp = int(bit_length(values.max()))
        for split_exp in range(min(max_exp, MAX_SPLIT_EXP) + 1):
            for msb_bits in range(min(split_exp, MAX_MSB_BITS) + 1):
                tokens, _, num_raw = tokenize(values, split_exp, msb_bits)
                counts = np.bincount(tokens)
                raw_bits = int(num_raw.sum())
                header_bits = 4 + 2
                for mantissa_bits in range(MAX_MANTISSA_BITS + 1):
                    codes = quantize_counts(counts, mantissa_bits)
                    freqs = normalize(code_weights(codes, mantissa_bits))
                    used = counts > 0
                    payload_bits = float((counts[used] * (PREC - np.log2(freqs[used]))).sum())
                    bits = header_bits + int(ue_bits(len(codes) - 1)) + 2 + table_bits(codes)[0] + payload_bits + raw_bits
                    if best is None or bits < best[-1]:
                        best = (split_exp, msb_bits, mantissa_bits, codes, payload_bits + raw_bits, bits)
        return cls(*best)

    def write(self, writer):
        writer.write(self.split_exp, 4)
        writer.write(self.msb_bits, 2)
        write_table(writer, self.codes, self.mantissa_bits)

    @classmethod
    def read(cls, reader):
        model = cls.__new__(cls)
        model.split_exp, model.msb_bits = reader.read(4), reader.read(2)
        model.freqs = read_table(reader)
        return model

    def encode(self, encoder, values):
        tokens, raw, num_raw = tokenize(values, self.split_exp, self.msb_bits)
        starts = np.concatenate([[0], np.cumsum(self.freqs)[:-1]])
        for token, start, freq, bits, count in zip(tokens.tolist(), starts[tokens].tolist(), self.freqs[tokens].tolist(),
                                                   raw.tolist(), num_raw.tolist()):
            encoder.put(start, freq, PREC)
            encoder.put_bits(bits, count)

    def decode(self, decoder, count):
        starts = np.concatenate([[0], np.cumsum(self.freqs)[:-1]])
        slot_tokens = np.repeat(np.arange(len(self.freqs)), self.freqs).tolist()
        starts, freqs = starts.tolist(), self.freqs.tolist()
        values = []
        for _ in range(count):
            token = slot_tokens[decoder.peek(PREC)]
            decoder.advance(starts[token], freqs[token], PREC)
            value, num_raw = detokenize(token, self.split_exp, self.msb_bits)
            values.append(value | decoder.get_bits(num_raw))
        return np.array(values, dtype=np.int64)


class RansEncoder:
    """
    Collects symbols in decoding order and encodes them in reverse, since rANS decodes in the opposite order of encoding.
    """

    def __init__(self):
        self.symbols = []

    def put(self, start, freq, prec):
        self.symbols.append((start, freq, prec))

    def put_bits(self, value, num_bits):
        while num_bits > 0:
            chunk = min(num_bits, RAW_CHUNK)
            num_bits -= chunk
            self.symbols.append(((value >> num_bits) & ((1 << chunk) - 1), 1, chunk))

    def finish(self):
        state, words = RANS_L, []
        for start, freq, prec in reversed(self.symbols):
            if state >= ((RANS_L >> prec) << 32) * freq:
                words.append(state & 0xFFFFFFFF)
                state >>= 32
            state = ((state // freq) << prec) + (state % freq) + start
        words += [state & 0xFFFFFFFF, state >> 32]
        return np.array(words[::-1], dtype="<u4").tobytes()


class RansDecoder:
    def __init__(self, data):
        self.words = np.frombuffer(data, dtype="<u4").tolist()
        self.state = (self.words[0] << 32) | self.words[1]
        self.pos = 2

    def peek(self, prec):
        return self.state & ((1 << prec) - 1)

    def advance(self, start, freq, prec):
        self.state = freq * (self.state >> prec) + (self.state & ((1 << prec) - 1)) - start
        if self.state < RANS_L:
            self.state = (self.state << 32) | self.words[self.pos]
            self.pos += 1

    def get_bits(self, num_bits):
        value = 0
        while num_bits > 0:
            chunk = min(num_bits, RAW_CHUNK)
            num_bits -= chunk
            bits = self.peek(chunk)
            self.advance(bits, 1, chunk)
            value = (value << chunk) | bits
        return value


def _fit_stream(values, cache):
    """
    Cheapest mapping of a stream of integers [int64] to non-negative ones (shift by its minimum or zigzag around its median)
    and its token model, with the total size in bits.
    """
    key = values.tobytes()
    if key not in cache:
        candidates = []
        for zigzag_mapped, param in [(0, int(values.min())), (1, int(np.round(np.median(values))))]:
            mapped = zigzag(values - param) if zigzag_mapped else values - param
            model = StreamModel.fit(mapped)
            candidates.append((1 + int(ue_bits(zigzag(np.array([param])))[0]) + model.bits, zigzag_mapped, param, mapped, model))
        cache[key] = min(candidates, key=lambda candidate: candidate[0])
    return cache[key]


def estimate_stream_bits(values):
    """
    Fast estimate of the coded size in bits of a stream of integers [int64], for rate decisions during optimization: the cheaper of the raw
    values and their differences, with ideal code lengths of their tokens (without most significant bits), the raw bits, and an approximate
    header cost per used token. Within about 3% of the size with fitted models (480p models, 450 to 3600 Gaussians) at a fraction of the time.
    """
    best = np.inf
    for mapped in [values - values.min(), zigzag(np.diff(values, prepend=0))]:
        for split_exp in range(min(int(bit_length(mapped.max())), MAX_SPLIT_EXP) + 1):
            tokens, _, num_raw = tokenize(mapped, split_exp, 0)
            counts = np.bincount(tokens)
            counts = counts[counts > 0]
            bits = float((counts * np.log2(len(mapped) / counts)).sum()) + float(num_raw.sum()) + TABLE_BITS_PER_TOKEN * len(counts)
            best = min(best, bits)
    return best


def _write_stream(writer, encoder, zigzag_mapped, param, mapped, model):
    writer.write(zigzag_mapped, 1)
    writer.write_se(param)
    model.write(writer)
    model.encode(encoder, mapped)


def _read_stream(reader):
    zigzag_mapped, param = reader.read(1), reader.read_se()
    return zigzag_mapped, param, StreamModel.read(reader)


def _decode_stream(decoder, stream, count):
    zigzag_mapped, param, model = stream
    mapped = model.decode(decoder, count)
    return (unzigzag(mapped) if zigzag_mapped else mapped) + param


def encode_streams(groups, writer):
    """
    Encode groups (attributes) of integer streams [P] int64. Each stream is coded as raw values or as differences between consecutive
    Gaussians, whichever is smaller. All streams share a single rANS state, and their models (token histograms) are written to writer.
    Returns the payload and the estimated size in bits of each group (models and payload).
    """
    encoder, cache, group_bits = RansEncoder(), {}, []
    for streams in groups:
        num_bits, payload_bits = writer.num_bits, 0.0
        for stream in streams:
            fits = [(_fit_stream(values, cache), delta) for delta, values in enumerate([stream, np.diff(stream, prepend=0)])]
            (_, zigzag_mapped, param, mapped, model), delta = min(fits, key=lambda fit: fit[0][0])
            writer.write(delta, 1)
            _write_stream(writer, encoder, zigzag_mapped, param, mapped, model)
            payload_bits += model.payload_bits
        group_bits.append(writer.num_bits - num_bits + payload_bits)
    return encoder.finish(), group_bits


def decode_streams(reader, data, num_streams, count):
    """
    Decode groups of num_streams[i] streams of count values, with the models read from reader and the payload following the header in data.
    """
    headers = [[(reader.read(1), _read_stream(reader)) for _ in range(num)] for num in num_streams]
    decoder = RansDecoder(data[reader.byte_pos():])
    return [[np.cumsum(_decode_stream(decoder, stream, count)) if delta else _decode_stream(decoder, stream, count) for delta, stream in streams]
            for streams in headers]
