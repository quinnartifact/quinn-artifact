// QUINN: Per-query latency timestamp shared memory accessor.
//
// Layout (created by Python LatencyShmWriter in shm.py):
//   Header  (16B): magic(4B) version(4B) num_queries(4B) pad(4B)
//   Entries       : LatencyEntry[num_queries]  (32B each)
//     spann_start_ns    int64   — SPANN writes before SearchIndex
//     spann_end_ns      int64   — SPANN writes after  SearchIndex
//     diskann_start_ns  int64   — DiskANN writes before cached_beam_search
//     diskann_end_ns    int64   — DiskANN writes after  cached_beam_search
//
// All fields initialised to 0 (0 == "not recorded").
// Reads happen only after both processes exit, so no synchronisation needed.
#pragma once

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace quinn {

struct LatencyEntry {
    int64_t spann_start_ns   = 0;   // SPANN: wall-clock start (ns)
    int64_t spann_end_ns     = 0;   // SPANN: wall-clock end   (ns)
    int64_t diskann_start_ns = 0;   // DiskANN: wall-clock start (ns)
    int64_t diskann_end_ns   = 0;   // DiskANN: wall-clock end (ns) — after full query (incl. CPU)
};
static_assert(sizeof(LatencyEntry) == 32, "LatencyEntry must be 32 bytes");

struct LatencyShmHeader {
    uint32_t magic;        // 0x4C415443 ('LATC')
    uint32_t version;      // 1
    uint32_t num_queries;
    uint32_t pad;
};
static_assert(sizeof(LatencyShmHeader) == 16, "LatencyShmHeader must be 16 bytes");

class LatencyShmAccessor {
public:
    static constexpr uint32_t MAGIC   = 0x4C415443;  // 'LATC'
    static constexpr uint32_t VERSION = 1;

    LatencyShmAccessor(const std::string &name, uint32_t num_queries)
        : _num_queries(num_queries),
          _total(sizeof(LatencyShmHeader) + static_cast<size_t>(num_queries) * sizeof(LatencyEntry))
    {
        int fd = ::shm_open(name.c_str(), O_RDWR, 0666);
        if (fd < 0)
            throw std::runtime_error("LatencyShmAccessor: shm_open failed: " + name);
        _ptr = ::mmap(nullptr, _total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ::close(fd);
        if (_ptr == MAP_FAILED)
            throw std::runtime_error("LatencyShmAccessor: mmap failed");
        _entries = reinterpret_cast<LatencyEntry *>(
            static_cast<char *>(_ptr) + sizeof(LatencyShmHeader));
    }

    ~LatencyShmAccessor() {
        if (_ptr && _ptr != MAP_FAILED)
            ::munmap(_ptr, _total);
    }

    // Convenience: current time in nanoseconds.
    static int64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    }

    void record_spann_start   (uint32_t qid, int64_t ns) { _entries[qid].spann_start_ns   = ns; }
    void record_spann_end     (uint32_t qid, int64_t ns) { _entries[qid].spann_end_ns     = ns; }
    void record_diskann_start (uint32_t qid, int64_t ns)  { _entries[qid].diskann_start_ns = ns; }
    void record_diskann_end   (uint32_t qid, int64_t ns)  { _entries[qid].diskann_end_ns   = ns; }

    const LatencyEntry &get(uint32_t qid) const { return _entries[qid]; }
    uint32_t num_queries() const { return _num_queries; }

private:
    uint32_t      _num_queries;
    size_t        _total;
    void         *_ptr     = nullptr;
    LatencyEntry *_entries = nullptr;
};

}  // namespace quinn
