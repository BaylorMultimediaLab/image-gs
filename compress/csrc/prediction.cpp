/*
 * Lossless prediction of quantized feature indices in bitstream order, for compress/prediction.py: the k nearest earlier Gaussians of each
 * Gaussian, and per channel the predicted index from their fixed-point values: their mean, or with a reference channel the least-squares
 * line between the channel and the reference over the neighbors (CCLM-style), evaluated at the Gaussian's own reference value. The sums
 * are integers and the rest is a fixed sequence of IEEE double operations on exactly representable values (built with -ffp-contract=off),
 * so that encoders and decoders compute the same predictions. Also the squared distances to the k nearest Gaussians, from which
 * compress/scale_prediction.py predicts the scales.
 * Build: pip install compress/csrc --no-build-isolation (from the repository root)
 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

#ifndef SOURCE_HASH
#define SOURCE_HASH "unknown"
#endif

namespace py = pybind11;

namespace {

constexpr int NUM_NEIGHBORS = 8;
constexpr int64_t MIN_CCLM_NEIGHBORS = 3;  // Fewer neighbors predict from their mean
constexpr double MAX_SLOPE = 4.0;
constexpr double POINTS_PER_CELL = 2.0;   // Of the search grid
constexpr int64_t QUERIES_PER_TASK = 256;  // Gaussians per task of the neighbor search
constexpr int64_t MIN_QUERIES_PER_THREAD = 1024;  // Fewer Gaussians per thread are searched by fewer threads

using IntArray = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

// Neighbor candidate, ordered by squared distance and then by index
struct Candidate {
    int64_t dist2;
    int64_t index;
    bool operator<(const Candidate& other) const {
        return dist2 < other.dist2 || (dist2 == other.dist2 && index < other.index);
    }
};

// Threads of the neighbor search: the CPUs this process may run on
int available_threads() {
#ifdef __linux__
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        return std::max(1, CPU_COUNT(&set));
    }
#endif
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
}

// The NUM_NEIGHBORS nearest earlier Gaussians of each Gaussian (integer positions [P, 2] in bitstream order; with causal false, the nearest
// other Gaussians), by squared distance and then by index, -1 where there are fewer: their indices and squared distances [P, NUM_NEIGHBORS].
// A uniform grid holds all Gaussians, each cell in increasing index order, and square rings of cells around a Gaussian are searched (each
// cell up to the Gaussian's own index when causal) until no cell of the next ring can hold a nearer one (the ring at Chebyshev cell distance
// r >= 1 is at least (r - 1) * cell + 1 away). The Gaussians are searched in parallel in tasks of QUERIES_PER_TASK; each one's result
// depends only on the positions.
void nearest_neighbors(const int64_t* positions, int64_t num, bool causal, std::vector<int64_t>& neighbors, std::vector<int64_t>& dist2) {
    neighbors.assign(num * NUM_NEIGHBORS, -1);
    dist2.assign(num * NUM_NEIGHBORS, -1);
    if (num == 0) {
        return;
    }
    int64_t min_x = positions[0], max_x = positions[0], min_y = positions[1], max_y = positions[1];
    for (int64_t i = 1; i < num; ++i) {
        min_x = std::min(min_x, positions[2 * i]);
        max_x = std::max(max_x, positions[2 * i]);
        min_y = std::min(min_y, positions[2 * i + 1]);
        max_y = std::max(max_y, positions[2 * i + 1]);
    }
    const double area = static_cast<double>(max_x - min_x + 1) * static_cast<double>(max_y - min_y + 1);
    const int64_t cell = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(std::sqrt(POINTS_PER_CELL * area / static_cast<double>(num)))));
    const int64_t grid_w = (max_x - min_x) / cell + 1, grid_h = (max_y - min_y) / cell + 1;
    // Members of each cell (counting sort by cell, in increasing index order): members[cell_start[c]:cell_start[c + 1]]
    std::vector<int64_t> cell_of(num), cell_start(grid_w * grid_h + 1, 0), members(num);
    for (int64_t i = 0; i < num; ++i) {
        cell_of[i] = (positions[2 * i + 1] - min_y) / cell * grid_w + (positions[2 * i] - min_x) / cell;
        ++cell_start[cell_of[i] + 1];
    }
    for (int64_t c = 0; c < grid_w * grid_h; ++c) {
        cell_start[c + 1] += cell_start[c];
    }
    std::vector<int64_t> fill(cell_start.begin(), cell_start.end() - 1);
    for (int64_t i = 0; i < num; ++i) {
        members[fill[cell_of[i]]++] = i;
    }
    auto search = [&](int64_t i, std::vector<Candidate>& best) {
        const int64_t x = positions[2 * i], y = positions[2 * i + 1];
        const int64_t cx = (x - min_x) / cell, cy = (y - min_y) / cell;
        best.clear();
        const int64_t max_ring = std::max({cx, grid_w - 1 - cx, cy, grid_h - 1 - cy});
        for (int64_t ring = 0; ring <= max_ring; ++ring) {
            if (static_cast<int64_t>(best.size()) == NUM_NEIGHBORS && ring > 0) {
                const int64_t gap = (ring - 1) * cell + 1;
                if (gap * gap > best.back().dist2) {
                    break;
                }
            }
            for (int64_t gy = std::max<int64_t>(0, cy - ring); gy <= std::min(grid_h - 1, cy + ring); ++gy) {
                const bool edge_row = gy == cy - ring || gy == cy + ring;
                const int64_t step = edge_row ? 1 : 2 * ring;  // Inner rows of the ring: only its left and right cells
                for (int64_t gx = cx - ring; gx <= cx + ring; gx += std::max<int64_t>(step, 1)) {
                    if (gx < 0 || gx >= grid_w) {
                        continue;
                    }
                    const int64_t c = gy * grid_w + gx;
                    for (int64_t m = cell_start[c]; m < cell_start[c + 1] && (!causal || members[m] < i); ++m) {
                        const int64_t j = members[m];
                        if (j == i) {
                            continue;
                        }
                        const int64_t dx = positions[2 * j] - x, dy = positions[2 * j + 1] - y;
                        const Candidate candidate{dx * dx + dy * dy, j};
                        if (static_cast<int64_t>(best.size()) < NUM_NEIGHBORS || candidate < best.back()) {
                            best.insert(std::upper_bound(best.begin(), best.end(), candidate), candidate);
                            if (static_cast<int64_t>(best.size()) > NUM_NEIGHBORS) {
                                best.pop_back();
                            }
                        }
                    }
                }
            }
        }
        for (size_t n = 0; n < best.size(); ++n) {
            neighbors[i * NUM_NEIGHBORS + n] = best[n].index;
            dist2[i * NUM_NEIGHBORS + n] = best[n].dist2;
        }
    };
    std::atomic<int64_t> next_task{0};
    auto worker = [&]() {
        std::vector<Candidate> best;
        best.reserve(NUM_NEIGHBORS + 1);
        for (int64_t start = next_task.fetch_add(QUERIES_PER_TASK); start < num; start = next_task.fetch_add(QUERIES_PER_TASK)) {
            for (int64_t i = start; i < std::min(num, start + QUERIES_PER_TASK); ++i) {
                search(i, best);
            }
        }
    };
    const int64_t num_threads = std::max<int64_t>(1, std::min<int64_t>(available_threads(), num / MIN_QUERIES_PER_THREAD));
    std::vector<std::thread> threads;
    for (int64_t t = 1; t < num_threads; ++t) {
        threads.emplace_back(worker);
    }
    worker();
    for (std::thread& thread : threads) {
        thread.join();
    }
}

std::vector<int64_t> causal_neighbors(const int64_t* positions, int64_t num) {
    std::vector<int64_t> neighbors, dist2;
    nearest_neighbors(positions, num, true, neighbors, dist2);
    return neighbors;
}

// Predicted index of channel target of a Gaussian with the fixed-point step step, from the fixed-point values [P, C] of its neighbors (-1
// terminated) and, with a reference channel ref >= 0, its own reference value own_ref
int64_t predict(const int64_t* values, int64_t num_channels, const int64_t* neighbors, int64_t step, int64_t target, int64_t ref,
                int64_t own_ref) {
    int64_t count = 0, sum_t = 0, sum_r = 0, sum_tr = 0, sum_rr = 0;
    for (int n = 0; n < NUM_NEIGHBORS && neighbors[n] >= 0; ++n) {
        const int64_t t = values[neighbors[n] * num_channels + target];
        ++count;
        sum_t += t;
        if (ref >= 0) {
            const int64_t r = values[neighbors[n] * num_channels + ref];
            sum_r += r;
            sum_tr += t * r;
            sum_rr += r * r;
        }
    }
    if (count == 0) {
        return 0;
    }
    if (ref < 0 || count < MIN_CCLM_NEIGHBORS) {
        return static_cast<int64_t>(std::nearbyint(static_cast<double>(sum_t) / static_cast<double>(count * step)));
    }
    const int64_t cov = count * sum_tr - sum_t * sum_r;
    const int64_t var = count * sum_rr - sum_r * sum_r;
    const double slope = std::clamp(static_cast<double>(cov) / static_cast<double>(var + count * count * step * step), -MAX_SLOPE, MAX_SLOPE);
    const double line = static_cast<double>(sum_t) + slope * static_cast<double>(count * own_ref - sum_r);
    return static_cast<int64_t>(std::nearbyint(line / static_cast<double>(count) / static_cast<double>(step)));
}

// Channels in decoding order: those without a reference first, then each after its reference
std::vector<int64_t> channel_order(const std::vector<int64_t>& refs) {
    const int64_t num_channels = static_cast<int64_t>(refs.size());
    std::vector<int64_t> order;
    std::vector<bool> done(num_channels, false);
    while (static_cast<int64_t>(order.size()) < num_channels) {
        const size_t before = order.size();
        for (int64_t c = 0; c < num_channels; ++c) {
            if (!done[c] && (refs[c] == -1 || (refs[c] >= 0 && refs[c] < num_channels && done[refs[c]]))) {
                order.push_back(c);
                done[c] = true;
            }
        }
        if (order.size() == before) {
            throw std::invalid_argument("Prediction references must form chains that start at channels without a reference");
        }
    }
    return order;
}

void check_shapes(const IntArray& indices, const IntArray& steps, const std::vector<int64_t>& refs, const IntArray& positions) {
    if (indices.ndim() != 2 || steps.ndim() != 2 || positions.ndim() != 2 || steps.shape(0) != indices.shape(0) ||
        steps.shape(1) != indices.shape(1) || positions.shape(0) != indices.shape(0) || positions.shape(1) != 2 ||
        static_cast<int64_t>(refs.size()) != indices.shape(1)) {
        throw std::invalid_argument("Expected indices and steps [P, C], references [C], and positions [P, 2]");
    }
}

// Residuals [P, C] of the indices [P, C] (bitstream order) with fixed-point steps [P, C], prediction references [C], and positions [P, 2]
IntArray predict_residuals(const IntArray& indices, const IntArray& steps, const std::vector<int64_t>& refs, const IntArray& positions) {
    check_shapes(indices, steps, refs, positions);
    const int64_t num = indices.shape(0), num_channels = indices.shape(1);
    const int64_t* q = indices.data();
    const int64_t* s = steps.data();
    std::vector<int64_t> values(num * num_channels);
    for (int64_t k = 0; k < num * num_channels; ++k) {
        values[k] = q[k] * s[k];
    }
    IntArray residuals({num, num_channels});
    int64_t* out = residuals.mutable_data();
    {
        py::gil_scoped_release release;
        const std::vector<int64_t> neighbors = causal_neighbors(positions.data(), num);
        for (int64_t c : channel_order(refs)) {
            for (int64_t i = 0; i < num; ++i) {
                const int64_t k = i * num_channels + c;
                const int64_t own_ref = refs[c] >= 0 ? values[i * num_channels + refs[c]] : 0;
                out[k] = q[k] - predict(values.data(), num_channels, &neighbors[i * NUM_NEIGHBORS], s[k], c, refs[c], own_ref);
            }
        }
    }
    return residuals;
}

// Indices [P, C] from the residuals of predict_residuals with the same steps, references, and positions
IntArray reconstruct_indices(const IntArray& residuals, const IntArray& steps, const std::vector<int64_t>& refs, const IntArray& positions) {
    check_shapes(residuals, steps, refs, positions);
    const int64_t num = residuals.shape(0), num_channels = residuals.shape(1);
    const int64_t* r = residuals.data();
    const int64_t* s = steps.data();
    IntArray indices({num, num_channels});
    int64_t* out = indices.mutable_data();
    {
        py::gil_scoped_release release;
        const std::vector<int64_t> neighbors = causal_neighbors(positions.data(), num);
        std::vector<int64_t> values(num * num_channels, 0);
        for (int64_t c : channel_order(refs)) {
            for (int64_t i = 0; i < num; ++i) {
                const int64_t k = i * num_channels + c;
                const int64_t own_ref = refs[c] >= 0 ? values[i * num_channels + refs[c]] : 0;
                out[k] = r[k] + predict(values.data(), num_channels, &neighbors[i * NUM_NEIGHBORS], s[k], c, refs[c], own_ref);
                values[k] = out[k] * s[k];
            }
        }
    }
    return indices;
}

// The causal neighbors themselves (for the rate term, tests, and analysis)
IntArray neighbors_of(const IntArray& positions) {
    if (positions.ndim() != 2 || positions.shape(1) != 2) {
        throw std::invalid_argument("Expected positions [P, 2]");
    }
    const int64_t num = positions.shape(0);
    const std::vector<int64_t> neighbors = causal_neighbors(positions.data(), num);
    IntArray out({num, static_cast<int64_t>(NUM_NEIGHBORS)});
    std::copy(neighbors.begin(), neighbors.end(), out.mutable_data());
    return out;
}

// Squared distances [P, NUM_NEIGHBORS] to the nearest other Gaussians (any order of the positions [P, 2]), increasing, -1 where there are fewer
IntArray nearest_distances(const IntArray& positions) {
    if (positions.ndim() != 2 || positions.shape(1) != 2) {
        throw std::invalid_argument("Expected positions [P, 2]");
    }
    const int64_t num = positions.shape(0);
    std::vector<int64_t> neighbors, dist2;
    nearest_neighbors(positions.data(), num, false, neighbors, dist2);
    IntArray out({num, static_cast<int64_t>(NUM_NEIGHBORS)});
    std::copy(dist2.begin(), dist2.end(), out.mutable_data());
    return out;
}

}  // namespace

PYBIND11_MODULE(igs_prediction, module) {
    module.doc() = "Lossless prediction of quantized feature indices (see compress/prediction.py)";
    module.attr("SOURCE_HASH") = SOURCE_HASH;
    module.attr("NUM_NEIGHBORS") = NUM_NEIGHBORS;
    module.def("predict_residuals", &predict_residuals, py::arg("indices"), py::arg("steps"), py::arg("refs"), py::arg("positions"));
    module.def("reconstruct_indices", &reconstruct_indices, py::arg("residuals"), py::arg("steps"), py::arg("refs"), py::arg("positions"));
    module.def("causal_neighbors", &neighbors_of, py::arg("positions"));
    module.def("nearest_distances", &nearest_distances, py::arg("positions"));
}
