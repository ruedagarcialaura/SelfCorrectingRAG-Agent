# FAISS Architecture & Usage Documentation

> **Version:** 1.14.1
> **Standard:** C++17 (C++20 when cuVS is disabled)
> **License:** MIT

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [Core Architecture](#3-core-architecture)
4. [Core Concepts](#4-core-concepts)
5. [Index Types Reference](#5-index-types-reference)
6. [Memory Model](#6-memory-model)
7. [Storage & Persistence](#7-storage--persistence)
8. [SIMD & Performance](#8-simd--performance)
9. [Build & Installation](#9-build--installation)
10. [Usage Guide](#10-usage-guide)
11. [Index Factory](#11-index-factory)
12. [GPU Support](#12-gpu-support)
13. [Meta-Indexes](#13-meta-indexes)
14. [Quantization Deep Dive](#14-quantization-deep-dive)
15. [Clustering](#15-clustering)
16. [Limitations & Scalability](#16-limitations--scalability)
17. [Execution Environment Assumptions](#17-execution-environment-assumptions)
18. [Integration Points](#18-integration-points)
19. [Key Takeaways for Integration](#19-key-takeaways-for-integration)

---

## 1. Overview

FAISS (Facebook AI Similarity Search) is a C++ library for efficient similarity search and clustering of dense vectors. It is designed for scenarios where the dataset is too large for exact brute-force search but must still be queried with high recall.

**Design goals:**
- Maximum single-node throughput for nearest-neighbor queries
- Flexible trade-off between accuracy, memory, and latency
- GPU acceleration as a drop-in replacement for CPU indexes
- Support for both exact and approximate nearest-neighbor (ANN) search

**Typical workloads:**
- Billion-scale approximate nearest-neighbor (ANN) search
- In-memory clustering of high-dimensional vectors
- Compressed vector storage (PQ, SQ) for RAM-constrained environments
- GPU-accelerated batch queries for retrieval pipelines

---

## 2. Repository Structure

```
faiss/                        # Top-level repo root
├── CMakeLists.txt            # Build system entry point (version 1.14.1)
├── INSTALL.md                # Installation guide
├── CHANGELOG.md              # Version history
├── benchs/                   # Benchmark scripts
├── c_api/                    # C language bindings
├── cmake/                    # CMake helper modules
├── conda/                    # Conda packaging
├── contrib/                  # Community-contributed extras
├── demos/                    # Example programs
├── misc/                     # Miscellaneous tools
├── perf_tests/               # Performance test suite
├── tests/                    # Unit & integration tests
├── tutorial/                 # Step-by-step tutorials (C++ and Python)
└── faiss/                    # Main library source
    ├── Index.h               # Abstract base class
    ├── IndexFlat.h/cpp       # Brute-force exact search
    ├── IndexIVF.h/cpp        # Inverted file structure
    ├── IndexHNSW.h/cpp       # Graph-based ANN (HNSW)
    ├── IndexPQ.h/cpp         # Product quantization
    ├── IndexBinary.h/cpp     # Binary vector indexes
    ├── VectorTransform.h/cpp # PCA, OPQ, random rotations
    ├── Clustering.h/cpp      # k-means and variants
    ├── MetaIndexes.h/cpp     # Wrapper/meta indexes
    ├── index_factory.h/cpp   # String-based construction
    ├── index_io.h/cpp        # read_index / write_index
    ├── impl/                 # Core building blocks
    │   ├── ProductQuantizer.h/cpp
    │   ├── ScalarQuantizer.h/cpp
    │   ├── AdditiveQuantizer.h/cpp
    │   ├── ResidualQuantizer.h/cpp
    │   ├── HNSW.h/cpp
    │   ├── NSG.h/cpp
    │   ├── NNDescent.h/cpp
    │   ├── io.h              # IOReader / IOWriter
    │   ├── mapped_io.h       # MappedFileIOReader (mmap)
    │   ├── zerocopy_io.h     # Zero-copy IOReader
    │   ├── maybe_owned_vector.h
    │   ├── index_read.cpp    # Deserialization
    │   ├── index_write.cpp   # Serialization
    │   └── simd_dispatch.h   # Runtime SIMD selection
    ├── invlists/             # Inverted list backends
    │   ├── InvertedLists.h/cpp
    │   ├── ArrayInvertedLists.h/cpp
    │   ├── BlockInvertedLists.h/cpp
    │   └── OnDiskInvertedLists.h/cpp
    ├── utils/                # Distances, heaps, sorting
    ├── gpu/                  # GPU index implementations
    │   ├── GpuResources.h
    │   ├── GpuIndex.h
    │   ├── GpuIndexFlat.h/cpp
    │   ├── GpuIndexIVFFlat.h/cpp
    │   ├── GpuIndexIVFPQ.h/cpp
    │   ├── GpuIndexCagra.h/cpp
    │   └── StandardGpuResources.h/cpp
    └── python/               # Python (SWIG) bindings
        ├── __init__.py
        ├── loader.py
        └── class_wrappers.py
```

---

## 3. Core Architecture

### 3.1 Index Class Hierarchy

All CPU indexes inherit from `faiss::Index` defined in `faiss/Index.h`. This abstract base provides a common polymorphic interface that allows all index types to be used interchangeably.

```
faiss::Index (abstract)
├── IndexFlatCodes
│   ├── IndexFlat              — brute-force, stores raw floats
│   │   ├── IndexFlatL2        — L2 distance, optional norm cache
│   │   ├── IndexFlatIP        — inner product distance
│   │   ├── IndexFlat1D        — 1-D sorted exact search
│   │   └── IndexFlatPanorama  — panoramic variant
│   ├── IndexPQ                — product quantization
│   └── IndexPQFastScan        — SIMD-accelerated PQ scan
│
├── IndexIVFInterface (+ Level1Quantizer)
│   └── IndexIVF               — inverted file base
│       ├── IndexIVFFlat
│       ├── IndexIVFPQ
│       ├── IndexIVFScalarQuantizer
│       ├── IndexIVFFastScan
│       └── IndexIVFAdditiveQuantizer (+ variants)
│
├── IndexHNSW                  — navigable small-world graph
│   ├── IndexHNSWFlat
│   ├── IndexHNSWPQ
│   ├── IndexHNSWSQ
│   ├── IndexHNSW2Level
│   └── IndexHNSWCagra
│
├── IndexPreTransform          — preprocessing wrapper
├── IndexIDMap                 — custom ID mapping
├── IndexIDMap2                — + reconstruct support
├── IndexShards                — horizontal sharding
├── IndexReplicas              — read replicas
├── IndexRefine                — reranking over another index
└── IndexSplitVectors          — split across dimension slices

faiss::IndexBinary (abstract, separate hierarchy)
├── IndexBinaryFlat
├── IndexBinaryIVF
├── IndexBinaryHNSW
├── IndexBinaryHash
└── IndexBinaryMultiHash
```

### 3.2 Binary Index Hierarchy

Binary indexes form a separate hierarchy because they operate on `uint8_t` vectors and use Hamming distance (`int32_t`) instead of float distances:

```cpp
struct IndexBinary {
    using component_t = uint8_t;
    using distance_t  = int32_t;
    int d;           // dimension in bits
    int code_size;   // d / 8 bytes per vector
};
```

### 3.3 Polymorphism Model

The core interface from `Index` (simplified):

```cpp
struct Index {
    int d;               // vector dimension
    idx_t ntotal;        // number of indexed vectors
    bool verbose;
    bool is_trained;
    MetricType metric_type;
    float metric_arg;

    // Training (required for IVF, PQ, HNSW; no-op for Flat)
    virtual void train(idx_t n, const float* x);

    // Adding vectors
    virtual void add(idx_t n, const float* x);
    virtual void add_with_ids(idx_t n, const float* x, const idx_t* xids);

    // Search
    virtual void search(
        idx_t n, const float* x, idx_t k,
        float* distances, idx_t* labels,
        const SearchParameters* params = nullptr) = 0;

    // Range search
    virtual void range_search(
        idx_t n, const float* x, float radius,
        RangeSearchResult* result,
        const SearchParameters* params = nullptr);

    // Utilities
    virtual void reset();
    virtual void reconstruct(idx_t key, float* recons);
    virtual idx_t remove_ids(const IDSelector& sel);

    // Codec interface (for compressed indexes)
    virtual size_t sa_code_size() const;
    virtual void sa_encode(idx_t n, const float* x, uint8_t* bytes) const;
    virtual void sa_decode(idx_t n, const uint8_t* bytes, float* x) const;
};
```

---

## 4. Core Concepts

### 4.1 Vectors

FAISS operates on dense `float32` vectors. A dataset of `n` vectors of dimension `d` is represented as a contiguous row-major C array of size `n * d`:

```cpp
// n vectors, dimension d
float* xb = new float[n * d];  // database
float* xq = new float[nq * d]; // queries
```

### 4.2 Metric Types

```cpp
enum MetricType {
    METRIC_INNER_PRODUCT = 0,   // inner product (not normalized)
    METRIC_L2 = 1,              // squared L2 distance
    METRIC_L1,                  // L1 (Manhattan) distance
    METRIC_Linf,                // L-infinity distance
    METRIC_Lp,                  // Lp distance (metric_arg = p)
    METRIC_Canberra,
    METRIC_BrayCurtis,
    METRIC_JensenShannon,
    METRIC_Jaccard,
};
```

Cosine similarity is typically handled by normalizing vectors to unit length and then using `METRIC_INNER_PRODUCT`.

### 4.3 Indexing Strategies

FAISS indexes can be categorized by their search strategy:

| Strategy | Example Indexes | Trade-off |
|----------|----------------|-----------|
| Exact brute-force | `IndexFlat` | 100% recall, O(n) per query |
| Inverted file (IVF) | `IndexIVFFlat`, `IndexIVFPQ` | Sub-linear search via clustering |
| Graph-based | `IndexHNSW` | Fast ANN, high memory |
| Quantization | `IndexPQ`, `IndexSQ` | Compressed storage, lower memory |
| Combined | `IndexIVFPQ`, `IndexIVFSQ` | IVF partitioning + quantization |

### 4.4 Quantization Strategies

- **Product Quantization (PQ):** Splits a `d`-dimensional vector into `M` sub-vectors and quantizes each with `2^nbits` centroids. Typical: M=8, nbits=8 gives 8 bytes per vector.
- **Scalar Quantization (SQ):** Quantizes each dimension independently to `nbits` bits (4, 6, 8, or 16).
- **Additive Quantization (AQ):** Represents a vector as a sum of `M` codebook entries — higher quality but more expensive.
- **Residual Quantization (RQ):** Iterative AQ that quantizes residuals of previous levels.
- **RaBitQ:** Bit-level quantization optimized for memory efficiency.

---

## 5. Index Types Reference

### 5.1 IndexFlat — Exact Brute-Force

```cpp
#include <faiss/IndexFlat.h>

IndexFlatL2 index(d);     // L2 distance
IndexFlatIP index(d);     // Inner product
```

- Stores raw `float32` vectors; no compression.
- 100% recall by definition.
- `IndexFlatL2` optionally caches L2 norms for faster search.
- **When to use:** Small datasets (< 1M vectors), as a baseline, or as the coarse quantizer inside IVF.
- **Memory:** `n * d * 4` bytes.

### 5.2 IndexIVFFlat — IVF with Exact Sub-Search

```cpp
#include <faiss/IndexIVFFlat.h>

IndexFlatL2 quantizer(d);
IndexIVFFlat index(&quantizer, d, nlist, METRIC_L2);
index.nprobe = 10;  // lists to probe at search time
```

- Partitions vectors into `nlist` Voronoi cells using a coarse quantizer.
- At search time, probes `nprobe` cells and does exact search within them.
- **When to use:** Large datasets (1M–100M) where you want ~exact results.
- **Training required:** Yes — trains the coarse quantizer on representative data.
- **Parameters:** `nlist` (typically `4*sqrt(n)`), `nprobe` (accuracy/speed trade-off).

### 5.3 IndexIVFPQ — IVF with Product Quantization

```cpp
#include <faiss/IndexIVFPQ.h>

IndexFlatL2 quantizer(d);
IndexIVFPQ index(&quantizer, d, nlist, M, nbits);
// M: PQ subquantizers (d must be divisible by M)
// nbits: bits per subquantizer (typically 8)
```

- Combines IVF partitioning with PQ compression for compact storage.
- Each vector stored as `M * nbits / 8` bytes.
- **When to use:** 100M–1B scale, limited RAM, acceptable recall ~90%.
- **Training required:** Yes (coarse quantizer + PQ codebook).

### 5.4 IndexIVFFastScan — SIMD-Accelerated IVF

Similar to `IndexIVFPQ` but uses SIMD-accelerated lookup table scanning (4-bit PQ codes packed into registers). Up to 4× faster search on AVX2/AVX512 hardware.

### 5.5 IndexHNSW — Hierarchical Navigable Small World

```cpp
#include <faiss/IndexHNSW.h>

IndexHNSWFlat index(d, M);  // M: graph connectivity (default 32)
// No training needed, but add() is slower than IVF
```

- Graph structure with hierarchical layers; greedy routing during search.
- No training step required.
- Very fast search (sub-millisecond per query at high recall).
- **When to use:** Up to ~100M vectors where build time is acceptable and high recall is needed.
- **Memory:** Higher than IVF — stores graph links (~4*M floats per node) plus vectors.
- **Parameters:** `M` (graph width, default 32), `efConstruction` (build quality), `efSearch` (search quality).

Variants:
- `IndexHNSWFlat` — stores raw vectors
- `IndexHNSWPQ` — with PQ compression (lower memory, lower recall)
- `IndexHNSWSQ` — with scalar quantization
- `IndexHNSW2Level` — two-level hierarchical structure

### 5.6 IndexPQ — Standalone Product Quantization

```cpp
#include <faiss/IndexPQ.h>

IndexPQ index(d, M, nbits);
```

Exhaustive search over PQ-compressed codes using Asymmetric Distance Computation (ADC). Useful when vectors must be compressed but IVF overhead is undesirable.

**Search types:**
- `ST_PQ` — Standard asymmetric PQ search (default)
- `ST_HE` — Hamming distance on codes
- `ST_SDC` — Symmetric distance computation
- `ST_polysemous` — Combined HE + PQ (polysemous codes)

### 5.7 IndexScalarQuantizer (SQ)

```cpp
#include <faiss/IndexScalarQuantizer.h>

IndexScalarQuantizer index(d, ScalarQuantizer::QT_8bit);
// QT_4bit, QT_6bit, QT_8bit, QT_fp16, QT_bf16
```

Quantizes each float dimension to a fixed number of bits. Faster decode than PQ, lower compression ratio.

### 5.8 Binary Indexes

```cpp
#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>

IndexBinaryFlat index(d);  // d in bits (must be multiple of 8)
// Vectors are uint8_t[d/8], distances are int32_t Hamming distances
```

| Index | Description |
|-------|-------------|
| `IndexBinaryFlat` | Exact Hamming search |
| `IndexBinaryIVF` | IVF + Hamming search |
| `IndexBinaryHNSW` | Graph-based binary search |
| `IndexBinaryHash` | Hash-based binary search |

---

## 6. Memory Model

### 6.1 Float Storage

`IndexFlat` and similar uncompressed indexes store raw `float32` vectors. In `IndexFlatCodes`, vectors are stored as a flat byte buffer (`codes`):

```cpp
// Effective layout for IndexFlat:
// codes.size() == ntotal * d * sizeof(float)
float* raw = reinterpret_cast<float*>(index.codes.data());
```

`IndexFlatL2` adds an optional L2 norm cache:
```cpp
index.sync_l2norms();   // compute and cache L2 norms
index.clear_l2norms();  // release cache
```

### 6.2 MaybeOwnedVector

`faiss/impl/maybe_owned_vector.h` provides a flexible container that can either own its data (heap-allocated `std::vector`) or hold a non-owning view into external memory:

```cpp
template <typename T>
struct MaybeOwnedVector {
    bool is_owned;
    std::vector<T> owned_data;  // when is_owned == true
    T* view_data;               // when is_owned == false
    size_t view_size;
    std::shared_ptr<MaybeOwnedVectorOwner> owner;  // optional lifecycle tracking
    T* c_ptr;    // pointer to actual data (either mode)
    size_t c_size;
};
```

This enables zero-copy integration — external buffers (e.g., mmap regions) can be used as index storage without any copy.

### 6.3 Inverted List Storage

For IVF indexes, vectors are stored in inverted lists. Three backends are available:

| Class | File | Description |
|-------|------|-------------|
| `ArrayInvertedLists` | `invlists/ArrayInvertedLists.h` | In-memory, `std::vector` per list |
| `BlockInvertedLists` | `invlists/BlockInvertedLists.h` | Fixed-size block allocator |
| `OnDiskInvertedLists` | `invlists/OnDiskInvertedLists.h` | Memory-mapped file on disk |

All three implement the same `InvertedLists` interface, making them interchangeable:

```cpp
// Replace in-memory lists with on-disk
OnDiskInvertedLists* disk_ivl = new OnDiskInvertedLists(
    nlist, code_size, "/path/to/storage.invlists");
ivf_index->replace_invlists(disk_ivl, /*own=*/true);
```

### 6.4 InvertedLists Interface

```cpp
struct InvertedLists {
    size_t nlist;      // number of inverted lists
    size_t code_size;  // bytes per stored code

    // Read access (caller must release)
    virtual size_t list_size(size_t list_no) const = 0;
    virtual const uint8_t* get_codes(size_t list_no) const = 0;
    virtual const idx_t* get_ids(size_t list_no) const = 0;
    virtual void release_codes(size_t list_no, const uint8_t* codes) const;
    virtual void release_ids(size_t list_no, const idx_t* ids) const;

    // Write access
    virtual size_t add_entries(
        size_t list_no, size_t n_entry,
        const idx_t* ids, const uint8_t* code);
};
```

---

## 7. Storage & Persistence

### 7.1 IOReader / IOWriter

Serialization is built on abstract `IOReader` and `IOWriter` interfaces (`faiss/impl/io.h`):

```cpp
struct IOReader {
    std::string name;
    virtual size_t operator()(void* ptr, size_t size, size_t nitems) = 0;
    virtual int filedescriptor();  // optional, for mmap support
};

struct IOWriter {
    std::string name;
    virtual size_t operator()(const void* ptr, size_t size, size_t nitems) = 0;
    virtual int filedescriptor();
};
```

Concrete implementations:

| Class | Backend |
|-------|---------|
| `FileIOReader` / `FileIOWriter` | `FILE*` via C stdio |
| `VectorIOReader` / `VectorIOWriter` | `std::vector<uint8_t>` in memory |
| `BufferedIOReader` / `BufferedIOWriter` | Buffered wrapper around any reader/writer |
| `MappedFileIOReader` | `mmap()` — zero-copy file reading |
| Zero-copy reader (in `zerocopy_io.h`) | View into external `uint8_t*` buffer |

### 7.2 High-Level Serialization API

```cpp
#include <faiss/index_io.h>

// Write
write_index(index, "my_index.faiss");

// Read
faiss::Index* index = read_index("my_index.faiss");

// In-memory serialization
faiss::VectorIOWriter writer;
write_index(index, &writer);  // writer.data now holds the bytes

faiss::VectorIOReader reader;
reader.data = writer.data;
faiss::Index* restored = read_index(&reader);
```

### 7.3 Serialization Format

Each index type is identified by a FourCC tag encoded as a `uint32_t`. Magic numbers and version fields are validated at read time to detect corruption or version mismatches. The utility functions:

```cpp
uint32_t fourcc(const char sx[4]);       // "IxFl" → uint32_t
void fourcc_inv(uint32_t x, char str[5]); // inverse
std::string fourcc_inv_printable(uint32_t x);
```

Security note: deserialization validates magic numbers and version fields before proceeding (see `faiss/impl/index_read.cpp`).

### 7.4 Memory-Mapped I/O (mmap)

`MappedFileIOReader` maps the file into virtual address space. The index data is read from pages loaded on demand by the OS — no upfront allocation. This is the primary mechanism for loading very large indexes that exceed available RAM, used in conjunction with `OnDiskInvertedLists`.

### 7.5 OnDiskInvertedLists

For IVF indexes larger than RAM:

```cpp
#include <faiss/invlists/OnDiskInvertedLists.h>

OnDiskInvertedLists odil(nlist, code_size, "index.invlists");
// odil.ptr points to mmap region
// odil.read_only = false allows updates
```

**Internal layout:**
```
[codes list 0][codes list 1]...[ids list 0][ids list 1]...
```

Capacity is rounded up to powers of 2. A slot allocator handles fragmentation when lists grow. For read-only deployments (`read_only=true`), the compact form (size == capacity) is used for optimal mmap performance.

---

## 8. SIMD & Performance

### 8.1 Multi-Variant Build

FAISS is compiled into multiple library variants, each targeting a different SIMD ISA:

| Variant | Flag | ISA | Description |
|---------|------|-----|-------------|
| `faiss` (generic) | `generic` | SSE2 | Baseline, always available on x86 |
| `faiss_avx2` | `avx2` | AVX2 | 256-bit FMA, ~2× faster |
| `faiss_avx512` | `avx512` | AVX-512 | 512-bit FMA, ~4× faster |
| `faiss_avx512_spr` | `avx512_spr` | Sapphire Rapids | AMX + VNNI extensions |
| `faiss_sve` | `sve` | ARM SVE | Scalable Vector Extension |
| Single binary | `dd` | Runtime detect | Dispatches at runtime |

### 8.2 Runtime Dispatch (simd_dispatch.h)

`faiss/impl/simd_dispatch.h` provides compile-time and runtime dispatch:

```cpp
// Supported SIMD levels
enum class SIMDLevel {
    NONE,        // generic
    AVX2,
    AVX512,
    AVX512_SPR,
    ARM_NEON,
    ARM_SVE,
};

// Main dispatch: calls f with highest available SIMD level
template <typename LambdaType>
inline auto with_simd_level(LambdaType&& action);

// 256-bit dispatch: promotes AVX512 → AVX2, SVE → NEON
template <typename LambdaType>
inline auto with_simd_level_256bit(LambdaType&& action);
```

### 8.3 Python Runtime Loading

The Python loader (`faiss/python/loader.py`) automatically selects the best available shared library:

```python
# Loading priority (highest to lowest)
# 1. swigfaiss_avx512_spr
# 2. swigfaiss_avx512
# 3. swigfaiss_avx2
# 4. swigfaiss_sve
# 5. swigfaiss (generic)

# Override via environment variables:
# FAISS_OPT_LEVEL=avx2       — force a specific level
# FAISS_DISABLE_CPU_FEATURES=AVX512  — disable feature(s)
```

CPU feature detection uses `numpy.__cpu_features__`.

### 8.4 FastScan Kernels

`IndexIVFPQFastScan` and `IndexPQFastScan` use a 4-bit PQ encoding with SIMD-accelerated lookup table scanning. On AVX2/AVX512 hardware, this delivers up to 4× search throughput compared to standard PQ scanning. The kernel sources live in `faiss/impl/pq_4bit/`.

---

## 9. Build & Installation

### 9.1 CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `FAISS_ENABLE_GPU` | ON | Build GPU indexes (requires CUDA) |
| `FAISS_ENABLE_PYTHON` | ON | Build Python bindings (requires SWIG) |
| `FAISS_OPT_LEVEL` | `generic` | SIMD: `generic`, `avx2`, `avx512`, `avx512_spr`, `sve`, `dd` |
| `FAISS_ENABLE_CUVS` | OFF | NVIDIA cuVS GPU backends |
| `FAISS_ENABLE_ROCM` | OFF | AMD ROCm GPU support |
| `FAISS_ENABLE_C_API` | OFF | Build C API |
| `FAISS_ENABLE_SVS` | OFF | Intel SVS graph-based index |
| `BUILD_TESTING` | ON | Build C++ test suite |
| `BUILD_SHARED_LIBS` | OFF | Shared library instead of static |

### 9.2 Build Commands

```shell
# Configure (CPU-only, no Python, release build with AVX2)
cmake -B build . \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_OPT_LEVEL=avx2

# Build C++ library
make -C build -j faiss          # generic
make -C build -j faiss_avx2     # AVX2
make -C build -j faiss_avx512   # AVX512

# Build and install Python bindings
make -C build -j swigfaiss
(cd build/faiss/python && python setup.py install)

# Run Python tests
(cd build/faiss/python && python setup.py build)
PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" pytest tests/test_*.py

# Run a single Python test
PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" pytest tests/test_index.py

# Run C++ tests (requires -DBUILD_TESTING=ON)
make -C build test
./build/tests/test_ivfpq_indexing   # single C++ test
```

### 9.3 Python Installation via Conda (recommended)

```shell
# CPU only
conda install -c pytorch faiss-cpu

# GPU (CUDA)
conda install -c pytorch faiss-gpu
```

---

## 10. Usage Guide

### 10.1 C++ Usage

#### Basic Flat Index

```cpp
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

int d = 128;   // vector dimension
int n = 10000; // number of vectors

// Build index
faiss::IndexFlatL2 index(d);
index.add(n, xb);  // xb: float[n*d]

// Search: find k=10 nearest neighbors for nq queries
int k = 10, nq = 100;
std::vector<float> distances(nq * k);
std::vector<faiss::idx_t> labels(nq * k);
index.search(nq, xq, k, distances.data(), labels.data());

// Save and load
faiss::write_index(&index, "index.faiss");
faiss::Index* loaded = faiss::read_index("index.faiss");
delete loaded;
```

#### IVF Index (requires training)

```cpp
#include <faiss/IndexIVFFlat.h>

int nlist = 100;  // number of Voronoi cells
faiss::IndexFlatL2 quantizer(d);
faiss::IndexIVFFlat index(&quantizer, d, nlist, faiss::METRIC_L2);

// Train on representative data (use all or a sample)
index.train(n_train, xtrain);

// Add database vectors
index.add(n, xb);

// Search (nprobe controls recall/speed trade-off)
index.nprobe = 10;
index.search(nq, xq, k, distances.data(), labels.data());
```

#### HNSW Index

```cpp
#include <faiss/IndexHNSW.h>

int M = 32;  // graph connectivity
faiss::IndexHNSWFlat index(d, M);
index.hnsw.efConstruction = 40;  // build quality

index.add(n, xb);  // no training needed

index.hnsw.efSearch = 16;  // search quality
index.search(nq, xq, k, distances.data(), labels.data());
```

#### In-Memory Serialization

```cpp
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

// Serialize to bytes
faiss::VectorIOWriter writer;
faiss::write_index(&index, &writer);
std::vector<uint8_t>& bytes = writer.data;

// Deserialize from bytes
faiss::VectorIOReader reader;
reader.data = bytes;
faiss::Index* restored = faiss::read_index(&reader);
```

### 10.2 Python Usage

```python
import faiss
import numpy as np

d = 128
n = 10000
xb = np.random.rand(n, d).astype('float32')
xq = np.random.rand(100, d).astype('float32')

# Flat index
index = faiss.IndexFlatL2(d)
index.add(xb)
distances, labels = index.search(xq, k=10)

# IVF index
nlist = 100
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
index.train(xb)
index.add(xb)
index.nprobe = 10
distances, labels = index.search(xq, k=10)

# Save / load
faiss.write_index(index, "my_index.faiss")
index = faiss.read_index("my_index.faiss")

# In-memory serialization
buf = faiss.serialize_index(index)        # numpy uint8 array
index2 = faiss.deserialize_index(buf)
```

---

## 11. Index Factory

`faiss/index_factory.h` provides a string-based index construction API, enabling configuration-driven index creation:

```cpp
#include <faiss/index_factory.h>

faiss::Index* index = faiss::index_factory(d, "IVF256,PQ32");
```

### 11.1 Factory String Syntax

| String | Index type |
|--------|-----------|
| `"Flat"` | `IndexFlatL2` |
| `"IVF256,Flat"` | `IndexIVFFlat` with 256 lists |
| `"IVF256,PQ32"` | `IndexIVFPQ`, 256 lists, M=32 |
| `"IVF256,PQ32x8"` | `IndexIVFPQ`, 256 lists, M=32, nbits=8 |
| `"HNSW32"` | `IndexHNSWFlat` with M=32 |
| `"IVF256,SQ8"` | `IndexIVFScalarQuantizer` 8-bit |
| `"PQ32"` | `IndexPQ` M=32 |
| `"PQ32x8"` | `IndexPQ` M=32 nbits=8 |
| `"IDMap,Flat"` | `IndexIDMap` wrapping `IndexFlat` |
| `"OPQ32_128,IVF256,PQ32"` | OPQ pre-transform + IVF + PQ |
| `"PCA64,IVF256,Flat"` | PCA reduction to 64-D + IVF |

### 11.2 Prefix Transforms

Transforms before a comma apply dimensionality reduction or rotation:
- `PCA64` — reduce to 64 dimensions via PCA
- `OPQ32_128` — optimized PQ rotation to 128 dims with 32 subquantizers
- `L2norm` — normalize vectors to unit length

### 11.3 Metric

```cpp
faiss::Index* index = faiss::index_factory(
    d, "IVF256,PQ32", faiss::METRIC_INNER_PRODUCT);
```

Verbosity:
```cpp
faiss::index_factory_verbose = 1;  // global flag
```

---

## 12. GPU Support

### 12.1 GPU Index Hierarchy

```
faiss::gpu::GpuIndex (GPU abstract base)
├── GpuIndexFlat
│   ├── GpuIndexFlatL2
│   └── GpuIndexFlatIP
├── GpuIndexIVFFlat
├── GpuIndexIVFPQ
├── GpuIndexIVFScalarQuantizer
└── GpuIndexCagra   (CAGRA graph-based, requires cuVS)
```

GPU indexes are drop-in replacements for their CPU counterparts — they inherit the same `faiss::Index` interface.

### 12.2 GpuResources

```cpp
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>

faiss::gpu::StandardGpuResources res;

faiss::gpu::GpuIndexIVFFlatConfig config;
config.device = 0;  // GPU device ID

faiss::gpu::GpuIndexIVFFlat gpu_index(
    &res, d, nlist, faiss::METRIC_L2, config);
gpu_index.train(n, xb);
gpu_index.add(n, xb);
gpu_index.search(nq, xq, k, distances, labels);
```

**Allocation types (`AllocType`):**
- `FlatData` — vector storage for GpuIndexFlat
- `IVFLists` — inverted list storage
- `Quantizer` — PQ/SQ lookup tables
- `TemporaryMemoryBuffer` — scratch space reused across calls
- `TemporaryMemoryOverflow` — overflow when scratch space is exceeded

**Memory spaces (`MemorySpace`):**
- `Device` — `cudaMalloc/Free`
- `Unified` — `cudaMallocManaged` (accessible from CPU)
- `Temporary` — freed immediately after each call

### 12.3 CPU ↔ GPU Transfer

```cpp
#include <faiss/gpu/GpuClonerOptions.h>

// CPU → GPU
faiss::Index* cpu_index = faiss::read_index("index.faiss");
faiss::Index* gpu_index = faiss::gpu::index_cpu_to_gpu(
    &res, 0, cpu_index);

// GPU → CPU
faiss::Index* back_to_cpu = faiss::gpu::index_gpu_to_cpu(gpu_index);
```

### 12.4 CAGRA (Graph-Based GPU Index)

CAGRA is a high-performance graph-based ANN index for GPU, available when `FAISS_ENABLE_CUVS=ON`:

```cpp
#include <faiss/gpu/GpuIndexCagra.h>

faiss::gpu::GpuIndexCagraConfig config;
config.device = 0;
config.graph_degree = 64;
config.intermediate_graph_degree = 128;

faiss::gpu::GpuIndexCagra cagra(&res, d, faiss::METRIC_L2, config);
cagra.train(n, xb);   // builds the CAGRA graph
cagra.search(nq, xq, k, distances, labels);
```

`IndexHNSWCagra` provides a CPU HNSW index initialized from CAGRA's graph, enabling fast CPU serving after GPU-accelerated build.

---

## 13. Meta-Indexes

Meta-indexes wrap other indexes to add functionality without modifying the underlying implementation.

### 13.1 IndexIDMap — Custom Vector IDs

By default, FAISS assigns sequential IDs (0, 1, 2, ...). `IndexIDMap` maps custom external IDs:

```cpp
#include <faiss/MetaIndexes.h>

faiss::IndexFlatL2 base(d);
faiss::IndexIDMap index(&base);

std::vector<faiss::idx_t> ids = {100, 200, 300};
index.add_with_ids(3, xb, ids.data());
// Search returns the custom IDs (100, 200, 300)
```

`IndexIDMap2` additionally supports `reconstruct(id, vector)`.

### 13.2 IndexShards — Horizontal Partitioning

Distributes data across multiple sub-indexes (e.g., for multi-GPU or multi-threaded search):

```cpp
#include <faiss/MetaIndexes.h>

faiss::IndexShards index(d, /*threaded=*/true, /*successive_ids=*/true);
index.add_shard(shard0);
index.add_shard(shard1);
index.add(n, xb);  // distributed across shards
```

### 13.3 IndexReplicas — Read Replicas

Maintains identical copies of an index on multiple devices and routes search queries round-robin (useful for GPU replicas):

```cpp
faiss::IndexReplicas index(d, /*threaded=*/true);
index.add_replica(replica0);
index.add_replica(replica1);
```

### 13.4 IndexPreTransform — Input Transformation

Applies a `VectorTransform` to queries and database vectors before passing them to the underlying index:

```cpp
#include <faiss/VectorTransform.h>

faiss::PCAMatrix pca(d, 64);  // reduce 128-D → 64-D
pca.train(n_train, xtrain);

faiss::IndexFlatL2 sub_index(64);
faiss::IndexPreTransform index(&pca, &sub_index);
index.add(n, xb);  // applies PCA before storing
```

Available transforms:
- `PCAMatrix` — PCA dimensionality reduction
- `OPQMatrix` — Optimized Product Quantization rotation
- `RandomRotationMatrix` — Random orthonormal rotation
- `NormalizationTransform` — L2-normalize vectors
- `RemapDimensionsTransform` — Select/permute dimensions

### 13.5 IndexRefine — Two-Stage Reranking

Wraps a fast approximate index with an exact refinement step:

```cpp
#include <faiss/IndexRefine.h>

faiss::IndexIVFPQ approx(quantizer, d, nlist, M, nbits);
faiss::IndexRefine index(&approx, &flat_index);
index.k_factor = 4;  // retrieve 4×k, rerank to k
index.search(nq, xq, k, distances, labels);
```

### 13.6 IndexSplitVectors

Splits the vector dimension across sub-indexes — each sub-index handles a slice of dimensions:

```cpp
faiss::IndexSplitVectors index(d, /*threaded=*/false);
index.add_sub_index(sub0);  // handles dims [0, d/2)
index.add_sub_index(sub1);  // handles dims [d/2, d)
```

---

## 14. Quantization Deep Dive

### 14.1 ProductQuantizer

`faiss/impl/ProductQuantizer.h` — core PQ implementation:

```cpp
struct ProductQuantizer {
    size_t d;     // input dimension
    size_t M;     // number of sub-quantizers
    size_t nbits; // bits per sub-quantizer
    size_t dsub;  // d / M — sub-vector dimension
    size_t ksub;  // 2^nbits — number of centroids per sub-quantizer
    std::vector<float> centroids; // M * ksub * dsub
};
```

**Training:**
```cpp
pq.train(n, xtrain);  // k-means on each sub-space
```

**Encoding (asymmetric):**
- Queries: computed as full float vectors
- Database: each vector → M indices (`uint8_t` per index if nbits=8)

**Search with ADC (Asymmetric Distance Computation):**
1. Precompute `M × ksub` distance table for each query
2. For each database code, look up and sum M table entries
3. Total: `M` additions per database vector (vs `d` multiplications for exact)

### 14.2 ScalarQuantizer

`faiss/impl/ScalarQuantizer.h`:

```cpp
enum QuantizerType {
    QT_8bit,          // 1 byte per dimension
    QT_4bit,          // 0.5 bytes per dimension
    QT_8bit_uniform,  // uniform quantization
    QT_4bit_uniform,
    QT_fp16,          // half precision
    QT_bf16,          // bfloat16
    QT_6bit,          // 0.75 bytes per dimension
    QT_8bit_direct,   // direct cast
    QT_8bit_direct_signed,
};
```

Quantizes each dimension independently. Faster encode/decode than PQ, less compression.

### 14.3 AdditiveQuantizer

Represents each vector as a sum of `M` codebook entries from `M` separate codebooks. Unlike PQ, the codebooks are not constrained to act on independent sub-spaces:

```
x ≈ c_1[q_1] + c_2[q_2] + ... + c_M[q_M]
```

Higher quality approximation than PQ at the same code size, but O(k^M) or beam-search encoding. Subclasses:
- `ResidualQuantizer` — quantizes residuals iteratively
- `LocalSearchQuantizer` — local search optimization
- `ProductResidualQuantizer` — product + residual hybrid
- `ProductLocalSearchQuantizer` — product + local search hybrid

### 14.4 RaBitQ

`RaBitQuantizer` — single-bit per dimension quantization with rebalancing, targeting extreme compression. Used in `IndexRaBitQ` variants.

---

## 15. Clustering

`faiss/Clustering.h` provides k-means clustering as a standalone utility and as the basis for IVF coarse quantizer training.

### 15.1 ClusteringParameters

```cpp
struct ClusteringParameters {
    int niter = 25;                // k-means iterations
    int nredo = 1;                 // number of independent runs (best kept)
    bool verbose = false;
    bool spherical = false;        // normalize centroids to unit sphere
    bool int_centroids = false;    // round centroids to integers
    bool update_index = false;     // rebuild assignment index each iteration
    bool frozen_centroids = false; // fix centroids (only reassign)
    int min_points_per_centroid = 39;
    int max_points_per_centroid = 256;
    int seed = 1234;
    bool check_input_data_for_NaNs = true;
    bool use_faster_subsampling = false;
    ClusteringInitMethod init_method;  // RANDOM, KMEANS_PLUS_PLUS, AFK_MC2
    uint16_t afkmc2_chain_length = 50; // for AFK_MC2 init
};
```

### 15.2 Clustering API

```cpp
#include <faiss/Clustering.h>

faiss::Clustering clus(d, k);
clus.niter = 20;

faiss::IndexFlatL2 assign_index(d);
clus.train(n, xtrain, assign_index);

// Centroids are in clus.centroids (k * d floats)
```

### 15.3 Initialization Methods

- `RANDOM` — Uniformly random initial centroids (default)
- `KMEANS_PLUS_PLUS` — Sequential distance-weighted sampling
- `AFK_MC2` — Assumption-Free K-MC² (faster than k-means++ for large datasets)

### 15.4 Integration with IVF

When `IndexIVF::train()` is called, it internally runs `Clustering` on the training data to find `nlist` centroids that become the coarse quantizer. The `Level1Quantizer::clustering_index` can be a GPU index to accelerate this step.

---

## 16. Limitations & Scalability

### 16.1 Single-Node Architecture

FAISS has no built-in distributed mode. All data must fit within a single machine's address space (RAM + mmap). For distributed setups:
- Horizontal sharding must be implemented by the caller.
- `IndexShards` helps with multi-GPU/multi-thread fan-out, but not multi-machine.
- The RPC server in `contrib/` provides a basic distributed interface.

### 16.2 RAM Constraints

| Index type | Approx. memory per vector |
|-----------|--------------------------|
| IndexFlat (d=128) | 512 bytes |
| IndexIVFFlat | ~512 bytes (+ IVF overhead) |
| IndexIVFPQ (M=8) | ~8 bytes |
| IndexHNSW (M=32, d=128) | ~768 bytes (links + vectors) |
| IndexIVFPQ FastScan (M=16, 4-bit) | ~8 bytes |

For > 100M vectors at full precision, `OnDiskInvertedLists` with mmap is recommended.

### 16.3 Float vs Binary Separation

The `Index` and `IndexBinary` hierarchies are completely separate:
- Float indexes operate on `float32` vectors.
- Binary indexes operate on `uint8_t` bit-packed vectors with Hamming distance.
- There is no cross-hierarchy interoperability.

### 16.4 Training Data Requirements

- **IVF:** Requires at least `39 * nlist` training vectors (min_points_per_centroid × nlist).
- **PQ:** Requires at least `256 * M` training vectors for reliable codebooks.
- Training data should be representative of the query/database distribution.

### 16.5 Thread Safety

- Multiple threads can call `search()` concurrently on the same index.
- `add()` and `train()` are **not** thread-safe.
- `IndexShards(threaded=true)` and `IndexReplicas(threaded=true)` internally use thread pools.

---

## 17. Execution Environment Assumptions

FAISS makes the following assumptions about the execution environment:

1. **CPU architecture:** x86_64 (primary) or ARM. SIMD levels are auto-detected at runtime via the Python loader or compile-time via CMake flags.
2. **Memory:** Continuous virtual address space available. For indexes >RAM, OS-level mmap support is required (Linux/macOS; Windows via `MapViewOfFile`).
3. **Single node:** No network I/O; all indexes are in-process.
4. **Floating point:** IEEE 754 `float32`. All operations assume 32-bit floats unless a quantized codec is used.
5. **Vector layout:** Row-major (C-order), contiguous in memory. Non-contiguous arrays must be copied before passing to FAISS.
6. **Numpy interop:** Python bindings expect `numpy` arrays with `dtype=float32` and `order='C'` (contiguous). The bindings call `.contiguous()` internally only for some operations.

---

## 18. Integration Points

### 18.1 C++ Library

Link `libfaiss.a` (or `.so`), include headers from `faiss/`:

```cmake
find_package(faiss REQUIRED)
target_link_libraries(my_app PRIVATE faiss)
```

### 18.2 Python Bindings (numpy)

```python
import faiss
import numpy as np

# All array inputs must be float32 C-contiguous
xb = np.ascontiguousarray(xb, dtype='float32')
distances, labels = index.search(xb, k=10)
# labels: int64 array shape (nq, k), -1 = not found
# distances: float32 array shape (nq, k)
```

### 18.3 C API

`faiss/c_api/` provides a C-compatible interface for use from non-C++ languages:

```c
#include <faiss/c_api/faiss_c.h>

FaissIndexFlatL2* index;
faiss_IndexFlatL2_new_with_metric(&index, d, METRIC_L2);
faiss_Index_add((FaissIndex*)index, n, xb);
faiss_Index_search((FaissIndex*)index, nq, xq, k, distances, labels);
faiss_Index_free((FaissIndex*)index);
```

### 18.4 On-Disk Serving

For large-scale production serving without loading full index into RAM:

```cpp
// Load index structure
faiss::Index* index = faiss::read_index("index_structure.faiss",
                                         faiss::IO_FLAG_MMAP);

// Attach on-disk inverted lists
auto* ivf = dynamic_cast<faiss::IndexIVF*>(index);
auto* odil = new faiss::OnDiskInvertedLists(
    ivf->nlist, ivf->code_size, "lists.invlists");
ivf->replace_invlists(odil, /*own=*/true);
```

### 18.5 PyTorch Integration

Tensors must be converted to numpy arrays first:

```python
import torch, faiss, numpy as np

t = torch.randn(1000, 128)
xb = t.numpy().astype('float32')  # or .cpu().numpy()
index.add(xb)

q = torch.randn(10, 128).numpy().astype('float32')
distances, labels = index.search(q, 10)
```

For GPU tensors, use `t.cpu().numpy()` or pass the data pointer directly using FAISS C++ bindings.

### 18.6 contrib/ Utilities

`faiss/contrib/` contains higher-level Python utilities:

| Module | Description |
|--------|-------------|
| `contrib/datasets.py` | Dataset loading helpers |
| `contrib/inspect_tools.py` | Inspect index internals |
| `contrib/ondisk.py` | On-disk index building pipeline |
| `contrib/rpc.py` | Simple RPC server for distributed search |
| `contrib/evaluation.py` | Recall@k computation |
| `contrib/torch_utils.py` | PyTorch/FAISS interop utilities |

### 18.7 Range Search

Range search returns all vectors within distance `r` of each query:

```cpp
faiss::RangeSearchResult result(nq);
index.range_search(nq, xq, r, &result);
// result.lims[i]..result.lims[i+1] gives hits for query i
// result.labels[], result.distances[] hold hits
```

---

## 19. Key Takeaways for Integration

1. **Index selection:** Use `IndexFlatL2` for small datasets or exact ground truth; `IndexIVFPQ` or `IndexIVFFlat` for large datasets; `IndexHNSW` for high-recall ANN with no training overhead.

2. **Training:** IVF and PQ indexes require training. Train on a representative sample (ideally 30–100× nlist vectors). The coarse quantizer (`IndexFlatL2`) can be GPU-accelerated.

3. **Persistence:** Use `write_index` / `read_index` for file I/O. For in-memory transfer (e.g., network serialization), use `VectorIOWriter` / `VectorIOReader`. For very large indexes, use `OnDiskInvertedLists` + `MappedFileIOReader`.

4. **SIMD acceleration:** Build with `-DFAISS_OPT_LEVEL=avx2` on modern x86 hardware. The Python loader auto-selects the best available variant at runtime.

5. **GPU:** `GpuIndexIVFFlat` and `GpuIndexIVFPQ` provide 10–100× throughput vs CPU for large batch queries. Use `index_cpu_to_gpu()` to convert existing CPU indexes.

6. **IDs:** FAISS uses 0-based sequential IDs by default. Wrap with `IndexIDMap` to use custom external IDs.

7. **Recall tuning:** Increase `nprobe` (IVF) or `efSearch` (HNSW) to improve recall at the cost of latency. Use `IndexRefine` to rerank top candidates with exact distances.

8. **Memory:** `IndexIVFPQ` with `M=8` gives ~8 bytes/vector (vs 512 bytes for `IndexFlat` at d=128). Use `OnDiskInvertedLists` when the total dataset exceeds available RAM.

9. **Thread safety:** Search is thread-safe; mutation (add/train) is not. Use `IndexShards` or `IndexReplicas` to parallelize search across threads or GPUs.

10. **Factory strings:** Use `index_factory("IVF4096,PQ64")` for configuration-file-driven index construction. This is the recommended approach for production systems where the index type may need to change without code modifications.
