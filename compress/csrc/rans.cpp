/*
 * rANS coding of integer streams with static token histograms, in the format of compress/rans_codec.py (which writes and reads the stream
 * headers): the model search of the encoder, encoding, decoding, and the fast size estimate used for rate decisions during optimization.
 * The model search reproduces the earlier Python/numpy implementation exactly, including the order of numpy's pairwise summation of the
 * costs, so that both choose the same models and write identical bitstreams.
 * Build: pip install compress/csrc --no-build-isolation (from the repository root)
 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#ifndef SOURCE_HASH
#define SOURCE_HASH "unknown"
#endif

namespace py = pybind11;

namespace {

constexpr int PREC = 16;                 // Precision of the token models (frequencies sum to 2^PREC)
constexpr int RAW_CHUNK = 16;            // Maximum number of raw bits per rANS symbol
constexpr uint64_t RANS_L = 1ull << 31;  // Lower bound of the rANS state (64-bit state with 32-bit renormalization, as in ryg_rans' rans64)
constexpr int MAX_SPLIT_EXP = 12;
constexpr int MAX_MSB_BITS = 3;
constexpr int MAX_MANTISSA_BITS = 3;
constexpr int TABLE_BITS_PER_TOKEN = 6;  // Approximate header cost of each used token in estimate_stream_bits
constexpr int MAX_EXPONENT = 62;         // Of decoded values and count weights, which must fit in int64

using Ints = std::vector<int64_t>;
using IntArray = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

// Bit length of a non-negative integer
int bit_length(int64_t value) {
    return value == 0 ? 0 : 64 - __builtin_clzll(static_cast<uint64_t>(value));
}

// Length of the order-0 Exp-Golomb code of a non-negative integer
int64_t ue_bits(int64_t value) {
    return 2 * bit_length(value + 1) - 1;
}

// Map signed integers to non-negative ones: 0, -1, 1, -2, ... -> 0, 1, 2, 3, ...
int64_t zigzag(int64_t value) {
    return static_cast<int64_t>((static_cast<uint64_t>(value) << 1) ^ static_cast<uint64_t>(value >> 63));
}

// Sum in the order of numpy's pairwise summation (pairwise_sum in numpy/_core/src/umath/loops_utils.h.src)
double pairwise_sum(const double* values, size_t n) {
    if (n < 8) {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum += values[i];
        }
        return sum;
    }
    if (n <= 128) {
        double partial[8];
        std::copy(values, values + 8, partial);
        size_t i = 8;
        for (; i < n - n % 8; i += 8) {
            for (int j = 0; j < 8; ++j) {
                partial[j] += values[i + j];
            }
        }
        double sum = ((partial[0] + partial[1]) + (partial[2] + partial[3])) + ((partial[4] + partial[5]) + (partial[6] + partial[7]));
        for (; i < n; ++i) {
            sum += values[i];
        }
        return sum;
    }
    size_t half = n / 2;
    half -= half % 8;
    return pairwise_sum(values, half) + pairwise_sum(values + half, n - half);
}

struct Token {
    int64_t token;
    int64_t raw;  // Raw bits
    int num_raw;  // Number of raw bits
};

// Hybrid integer coding of a non-negative integer: values below 2^split_exp are their own token, larger values are coded by their bit length
// and their msb_bits most significant bits below the leading one, followed by the remaining bits as raw bits
Token tokenize(int64_t value, int split_exp, int msb_bits) {
    const int64_t split = int64_t(1) << split_exp;
    if (value < split) {
        return {value, 0, 0};
    }
    const int exponent = bit_length(value) - 1;
    const int num_raw = exponent - msb_bits;
    const int64_t msbs = (value >> num_raw) & ((int64_t(1) << msb_bits) - 1);
    return {split + (int64_t(exponent - split_exp) << msb_bits) + msbs, value & ((int64_t(1) << num_raw) - 1), num_raw};
}

// Value without its raw bits and number of raw bits of a token
std::pair<int64_t, int> detokenize(int64_t token, int split_exp, int msb_bits) {
    const int64_t split = int64_t(1) << split_exp;
    if (token < split) {
        return {token, 0};
    }
    const int num_raw = split_exp + static_cast<int>((token - split) >> msb_bits) - msb_bits;
    return {((int64_t(1) << msb_bits) | ((token - split) & ((int64_t(1) << msb_bits) - 1))) << num_raw, num_raw};
}

// Log-quantized code of a token count: 0 for unused tokens, otherwise (exponent << mantissa_bits) + mantissa, with weight mantissa << exponent
int64_t quantize_count(int64_t count, int mantissa_bits) {
    if (count < (int64_t(2) << mantissa_bits)) {
        return count;
    }
    int exponent = std::max(bit_length(count) - (mantissa_bits + 1), 0);
    int64_t mantissa = (count + (exponent > 0 ? int64_t(1) << std::max(exponent - 1, 0) : 0)) >> exponent;
    if (mantissa == (int64_t(2) << mantissa_bits)) {
        mantissa = int64_t(1) << mantissa_bits;
        ++exponent;
    }
    return (int64_t(exponent) << mantissa_bits) + mantissa;
}

int64_t code_weight(int64_t code, int mantissa_bits) {
    if (code < (int64_t(2) << mantissa_bits)) {
        return code;
    }
    const int64_t exponent = (code >> mantissa_bits) - 1;
    if (exponent > MAX_EXPONENT - PREC - (mantissa_bits + 1)) {
        throw py::value_error("Invalid token count code");
    }
    return (code - (exponent << mantissa_bits)) << exponent;
}

// Integer frequencies summing to 2^PREC, positive for the tokens with positive weight
void normalize(const Ints& weights, Ints& freqs) {
    int64_t total = 0;
    for (int64_t weight : weights) {
        total += weight;
    }
    if (total <= 0) {
        throw py::value_error("Cannot normalize the token histogram");
    }
    freqs.resize(weights.size());
    int64_t sum = 0;
    size_t largest = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        freqs[i] = weights[i] > 0 ? std::max<int64_t>((weights[i] << PREC) / total, 1) : 0;
        sum += freqs[i];
        if (freqs[i] > freqs[largest]) {
            largest = i;
        }
    }
    freqs[largest] += (int64_t(1) << PREC) - sum;
    if (freqs[largest] < 1) {
        throw py::value_error("Cannot normalize the token histogram");
    }
}

// Size in bits of a table of count codes after its length and mantissa_bits (write_table in compress/rans_codec.py)
int64_t table_bits(const Ints& codes) {
    const int64_t fixed = 5 + static_cast<int64_t>(codes.size()) * bit_length(*std::max_element(codes.begin(), codes.end()));
    int64_t delta = 0, previous = 0;
    for (int64_t code : codes) {
        delta += ue_bits(zigzag(code - previous));
        previous = code;
    }
    return 1 + std::min(fixed, delta);
}

// Code length in bits of a token with each frequency, PREC - log2(freq)
const std::vector<double>& code_lengths() {
    static const std::vector<double> lengths = [] {
        std::vector<double> lengths((size_t(1) << PREC) + 1, 0.0);
        for (size_t freq = 1; freq < lengths.size(); ++freq) {
            lengths[freq] = PREC - std::log2(static_cast<double>(freq));
        }
        return lengths;
    }();
    return lengths;
}

struct Model {
    int split_exp = 0, msb_bits = 0, mantissa_bits = 0;
    Ints codes, freqs;
    double payload_bits = 0.0;  // Estimated size of the tokens and raw bits
    double bits = 0.0;          // Estimated total size including the model
};

// Histogram of the tokens of non-negative integers with the finest tokenizer that any search considers: it refines every coarser tokenizer,
// whose tokens and numbers of raw bits follow from the smallest value of each fine token (detokenize)
Ints fine_histogram(const Ints& values, int split_exp, int msb_bits) {
    Ints counts;
    for (int64_t value : values) {
        const int64_t token = tokenize(value, split_exp, msb_bits).token;
        if (token >= static_cast<int64_t>(counts.size())) {
            counts.resize(token + 1, 0);
        }
        ++counts[token];
    }
    return counts;
}

// Token counts and total number of raw bits with a coarser tokenizer, aggregated from a fine histogram
int64_t aggregate(const Ints& fine_counts, int fine_split_exp, int fine_msb_bits, int split_exp, int msb_bits, Ints& counts) {
    counts.clear();
    int64_t raw_bits = 0;
    for (size_t fine = 0; fine < fine_counts.size(); ++fine) {
        if (fine_counts[fine] == 0) {
            continue;
        }
        const Token token = tokenize(detokenize(static_cast<int64_t>(fine), fine_split_exp, fine_msb_bits).first, split_exp, msb_bits);
        if (token.token >= static_cast<int64_t>(counts.size())) {
            counts.resize(token.token + 1, 0);
        }
        counts[token.token] += fine_counts[fine];
        raw_bits += fine_counts[fine] * token.num_raw;
    }
    return raw_bits;
}

// Token model of a stream of non-negative integers with the smallest total size (model and payload): all tokenizers and count quantizations
// in the order of the reference implementation, keeping the first of equal sizes
Model fit_model(const Ints& values) {
    const int max_split_exp = std::min(bit_length(*std::max_element(values.begin(), values.end())), MAX_SPLIT_EXP);
    const int fine_msb_bits = std::min(max_split_exp, MAX_MSB_BITS);
    const Ints fine_counts = fine_histogram(values, max_split_exp, fine_msb_bits);
    const std::vector<double>& lengths = code_lengths();
    Model best;
    bool found = false;
    Ints counts, codes, weights, freqs;
    std::vector<double> terms;
    for (int split_exp = 0; split_exp <= max_split_exp; ++split_exp) {
        for (int msb_bits = 0; msb_bits <= std::min(split_exp, MAX_MSB_BITS); ++msb_bits) {
            const int64_t raw_bits = aggregate(fine_counts, max_split_exp, fine_msb_bits, split_exp, msb_bits, counts);
            // Header: split_exp (4 bits), msb_bits (2), number of tokens, mantissa_bits (2), and the table (at least a bit per token)
            const int64_t fixed_header_bits = 4 + 2 + ue_bits(static_cast<int64_t>(counts.size()) - 1) + 2;
            if (found) {
                // Skip the tokenizer if it cannot beat the best model with any count quantization, which leaves the result unchanged: the
                // payload is at least the entropy of the token counts (Gibbs' inequality), with a margin far above rounding errors
                terms.clear();
                for (int64_t count : counts) {
                    if (count > 0) {
                        terms.push_back(static_cast<double>(count) * std::log2(static_cast<double>(values.size()) / static_cast<double>(count)));
                    }
                }
                const double lower_bound = static_cast<double>(fixed_header_bits + 1 + static_cast<int64_t>(counts.size())) +
                                           pairwise_sum(terms.data(), terms.size()) + static_cast<double>(raw_bits);
                if (lower_bound > best.bits * (1.0 + 1e-9) + 1e-6) {
                    continue;
                }
            }
            codes.resize(counts.size());
            weights.resize(counts.size());
            for (int mantissa_bits = 0; mantissa_bits <= MAX_MANTISSA_BITS; ++mantissa_bits) {
                for (size_t i = 0; i < counts.size(); ++i) {
                    codes[i] = quantize_count(counts[i], mantissa_bits);
                    weights[i] = code_weight(codes[i], mantissa_bits);
                }
                normalize(weights, freqs);
                terms.clear();
                for (size_t i = 0; i < counts.size(); ++i) {
                    if (counts[i] > 0) {
                        terms.push_back(static_cast<double>(counts[i]) * lengths[freqs[i]]);
                    }
                }
                const double payload_bits = pairwise_sum(terms.data(), terms.size());
                const double bits = static_cast<double>(fixed_header_bits + table_bits(codes)) + payload_bits + static_cast<double>(raw_bits);
                if (!found || bits < best.bits) {
                    found = true;
                    best = {split_exp, msb_bits, mantissa_bits, codes, freqs, payload_bits + static_cast<double>(raw_bits), bits};
                }
            }
        }
    }
    return best;
}

// Median of integers rounded to the nearest integer (ties to even), as int(np.round(np.median(values)))
int64_t rounded_median(Ints values) {
    const size_t half = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + half, values.end());
    double median = static_cast<double>(values[half]);
    if (values.size() % 2 == 0) {
        median = (static_cast<double>(*std::max_element(values.begin(), values.begin() + half)) + median) / 2.0;
    }
    return static_cast<int64_t>(std::nearbyint(median));
}

struct StreamFit {
    double bits = 0.0;  // Total size including the mapping
    bool zigzag_mapped = false;
    int64_t param = 0;
    Ints mapped;
    Model model;
};

// Cheapest mapping of a stream of integers to non-negative ones (shift by its minimum or zigzag around its median) and its token model
StreamFit fit_stream(const Ints& values) {
    StreamFit best;
    for (int zigzag_mapped = 0; zigzag_mapped < 2; ++zigzag_mapped) {
        const int64_t param = zigzag_mapped ? rounded_median(values) : *std::min_element(values.begin(), values.end());
        Ints mapped(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            mapped[i] = zigzag_mapped ? zigzag(values[i] - param) : values[i] - param;
        }
        Model model = fit_model(mapped);
        const double bits = static_cast<double>(1 + ue_bits(zigzag(param))) + model.bits;  // Mapping flag and parameter
        if (!zigzag_mapped || bits < best.bits) {
            best = {bits, zigzag_mapped == 1, param, std::move(mapped), std::move(model)};
        }
    }
    return best;
}

struct Symbol {
    uint32_t start, freq;
    int prec;
};

// rANS encoding of symbols given in decoding order: they are encoded in reverse, since rANS decodes in the opposite order of encoding
std::string rans_encode(const std::vector<Symbol>& symbols) {
    uint64_t state = RANS_L;
    std::vector<uint32_t> words;
    for (auto symbol = symbols.rbegin(); symbol != symbols.rend(); ++symbol) {
        const uint64_t freq = symbol->freq;
        if (state >= ((RANS_L >> symbol->prec) << 32) * freq) {
            words.push_back(static_cast<uint32_t>(state));
            state >>= 32;
        }
        state = ((state / freq) << symbol->prec) + state % freq + symbol->start;
    }
    words.push_back(static_cast<uint32_t>(state));
    words.push_back(static_cast<uint32_t>(state >> 32));
    std::string data(4 * words.size(), '\0');
    for (size_t i = 0; i < words.size(); ++i) {  // Little-endian 32-bit words, last emitted first
        const uint32_t word = words[words.size() - 1 - i];
        for (int byte = 0; byte < 4; ++byte) {
            data[4 * i + byte] = static_cast<char>((word >> (8 * byte)) & 0xFF);
        }
    }
    return data;
}

Ints to_ints(const IntArray& array) {
    if (array.ndim() != 1 || array.size() == 0) {
        throw py::value_error("Streams must be non-empty one-dimensional arrays");
    }
    return Ints(array.data(), array.data() + array.size());
}

// Fits the model of each stream (raw values or differences between consecutive values, whichever is smaller, then the mapping and the token
// model) and encodes all streams with a single rANS state. Returns the payload and per stream (delta, zigzag_mapped, param, split_exp,
// msb_bits, mantissa_bits, count codes, estimated payload bits).
py::tuple encode_streams(const std::vector<IntArray>& arrays) {
    std::vector<Ints> streams;
    for (const IntArray& array : arrays) {
        streams.push_back(to_ints(array));
    }
    std::vector<std::pair<bool, StreamFit>> fits;
    std::string payload;
    {
        py::gil_scoped_release release;
        // Fits of the raw values and of the differences of every stream, which are independent, on parallel threads
        std::vector<Ints> candidates;
        for (const Ints& stream : streams) {
            Ints deltas(stream.size());
            for (size_t i = 0; i < stream.size(); ++i) {
                deltas[i] = stream[i] - (i > 0 ? stream[i - 1] : 0);
            }
            candidates.push_back(stream);
            candidates.push_back(std::move(deltas));
        }
        std::vector<StreamFit> candidate_fits(candidates.size());
        std::atomic<size_t> next_task{0};
        std::exception_ptr error;
        std::mutex error_mutex;
        auto work = [&]() {
            for (size_t task = next_task++; task < candidates.size(); task = next_task++) {
                try {
                    candidate_fits[task] = fit_stream(candidates[task]);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    error = error ? error : std::current_exception();
                }
            }
        };
        std::vector<std::thread> threads;
        const size_t num_threads = std::min<size_t>(candidates.size(), std::max(1u, std::thread::hardware_concurrency()));
        for (size_t i = 1; i < num_threads; ++i) {
            threads.emplace_back(work);
        }
        work();
        for (std::thread& thread : threads) {
            thread.join();
        }
        if (error) {
            std::rethrow_exception(error);
        }
        std::vector<Symbol> symbols;
        for (size_t i = 0; i < streams.size(); ++i) {
            StreamFit& raw_fit = candidate_fits[2 * i];
            StreamFit& delta_fit = candidate_fits[2 * i + 1];
            const bool delta = delta_fit.bits < raw_fit.bits;
            fits.emplace_back(delta, std::move(delta ? delta_fit : raw_fit));
            const Model& model = fits.back().second.model;
            Ints starts(model.freqs.size(), 0);
            for (size_t i = 1; i < starts.size(); ++i) {
                starts[i] = starts[i - 1] + model.freqs[i - 1];
            }
            for (int64_t value : fits.back().second.mapped) {
                const Token token = tokenize(value, model.split_exp, model.msb_bits);
                symbols.push_back({static_cast<uint32_t>(starts[token.token]), static_cast<uint32_t>(model.freqs[token.token]), PREC});
                for (int remaining = token.num_raw; remaining > 0;) {  // Raw bits as uniform symbols, most significant first
                    const int chunk = std::min(remaining, RAW_CHUNK);
                    remaining -= chunk;
                    symbols.push_back({static_cast<uint32_t>((token.raw >> remaining) & ((int64_t(1) << chunk) - 1)), 1, chunk});
                }
            }
        }
        payload = rans_encode(symbols);
    }
    py::list models;
    for (const auto& [delta, fit] : fits) {
        const Model& model = fit.model;
        models.append(py::make_tuple(int(delta), int(fit.zigzag_mapped), fit.param, model.split_exp, model.msb_bits, model.mantissa_bits,
                                     py::array_t<int64_t>(model.codes.size(), model.codes.data()), model.payload_bits));
    }
    return py::make_tuple(py::bytes(payload), models);
}

struct DecodingModel {
    int split_exp, msb_bits;
    Ints freqs, starts;
    std::vector<uint32_t> tokens;  // Token of each of the 2^PREC slots
};

// Decodes streams of count values each from the payload, with their models (split_exp, msb_bits, mantissa_bits, count codes). Returns the
// mapped values [streams, count] (compress/rans_codec.py undoes the mapping and the differences).
IntArray decode_streams(const py::bytes& payload_bytes, const std::vector<std::tuple<int, int, int, IntArray>>& models, int64_t count) {
    const std::string payload = payload_bytes;
    if (payload.size() % 4 != 0 || payload.size() < 8 || count < 0) {
        throw py::value_error("Invalid rANS payload");
    }
    std::vector<uint32_t> words(payload.size() / 4);
    for (size_t i = 0; i < words.size(); ++i) {
        for (int byte = 0; byte < 4; ++byte) {
            words[i] |= uint32_t(static_cast<uint8_t>(payload[4 * i + byte])) << (8 * byte);
        }
    }
    std::vector<DecodingModel> decoding;
    for (const auto& [split_exp, msb_bits, mantissa_bits, codes_array] : models) {
        const Ints codes = to_ints(codes_array);
        if (split_exp < 0 || msb_bits < 0 || msb_bits > split_exp || mantissa_bits < 0) {
            throw py::value_error("Invalid token model");
        }
        const int64_t last = static_cast<int64_t>(codes.size()) - 1;  // Largest token, with the largest value
        if (last >= (int64_t(1) << split_exp) && split_exp + ((last - (int64_t(1) << split_exp)) >> msb_bits) > MAX_EXPONENT) {
            throw py::value_error("Invalid token model");
        }
        DecodingModel model{split_exp, msb_bits, {}, Ints(codes.size(), 0), {}};
        Ints weights(codes.size());
        for (size_t i = 0; i < codes.size(); ++i) {
            weights[i] = code_weight(codes[i], mantissa_bits);
        }
        normalize(weights, model.freqs);
        for (size_t token = 0; token < codes.size(); ++token) {
            if (token > 0) {
                model.starts[token] = model.starts[token - 1] + model.freqs[token - 1];
            }
            model.tokens.insert(model.tokens.end(), model.freqs[token], static_cast<uint32_t>(token));
        }
        decoding.push_back(std::move(model));
    }
    IntArray result({static_cast<py::ssize_t>(decoding.size()), static_cast<py::ssize_t>(count)});
    int64_t* values = result.mutable_data();
    {
        py::gil_scoped_release release;
        uint64_t state = (uint64_t(words[0]) << 32) | words[1];
        size_t pos = 2;
        auto renormalize = [&]() {
            if (state < RANS_L) {
                if (pos >= words.size()) {
                    throw py::value_error("Truncated rANS payload");
                }
                state = (state << 32) | words[pos++];
            }
        };
        for (const DecodingModel& model : decoding) {
            for (int64_t i = 0; i < count; ++i) {
                const uint64_t slot = state & ((uint64_t(1) << PREC) - 1);
                const uint32_t token = model.tokens[slot];
                state = model.freqs[token] * (state >> PREC) + slot - model.starts[token];
                renormalize();
                auto [value, num_raw] = detokenize(token, model.split_exp, model.msb_bits);
                int64_t raw = 0;
                for (int remaining = num_raw; remaining > 0;) {
                    const int chunk = std::min(remaining, RAW_CHUNK);
                    remaining -= chunk;
                    raw = (raw << chunk) | static_cast<int64_t>(state & ((uint64_t(1) << chunk) - 1));
                    state >>= chunk;
                    renormalize();
                }
                *values++ = value | raw;
            }
        }
    }
    return result;
}

// Fast estimate of the coded size in bits of a stream of integers (compress/rans_codec.py)
double estimate_stream_bits(const IntArray& array) {
    const Ints values = to_ints(array);
    py::gil_scoped_release release;
    const int64_t minimum = *std::min_element(values.begin(), values.end());
    Ints shifted(values.size()), deltas(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        shifted[i] = values[i] - minimum;
        deltas[i] = zigzag(values[i] - (i > 0 ? values[i - 1] : 0));
    }
    double best = std::numeric_limits<double>::infinity();
    Ints counts;
    std::vector<double> terms;
    for (const Ints* mapped : {&shifted, &deltas}) {
        const int max_split_exp = std::min(bit_length(*std::max_element(mapped->begin(), mapped->end())), MAX_SPLIT_EXP);
        const Ints fine_counts = fine_histogram(*mapped, max_split_exp, 0);
        for (int split_exp = 0; split_exp <= max_split_exp; ++split_exp) {
            const int64_t raw_bits = aggregate(fine_counts, max_split_exp, 0, split_exp, 0, counts);
            terms.clear();
            for (int64_t token_count : counts) {
                if (token_count > 0) {
                    const double n = static_cast<double>(token_count);
                    terms.push_back(n * std::log2(static_cast<double>(mapped->size()) / n));
                }
            }
            const double bits = pairwise_sum(terms.data(), terms.size()) + static_cast<double>(raw_bits) +
                                static_cast<double>(TABLE_BITS_PER_TOKEN * static_cast<int64_t>(terms.size()));
            best = std::min(best, bits);
        }
    }
    return best;
}

}  // namespace

PYBIND11_MODULE(igs_rans, module) {
    module.doc() = "rANS coding of integer streams with static token histograms (see compress/rans_codec.py)";
    module.attr("SOURCE_HASH") = SOURCE_HASH;
    module.def("encode_streams", &encode_streams, py::arg("streams"));
    module.def("decode_streams", &decode_streams, py::arg("payload"), py::arg("models"), py::arg("count"));
    module.def("estimate_stream_bits", &estimate_stream_bits, py::arg("values"));
}
