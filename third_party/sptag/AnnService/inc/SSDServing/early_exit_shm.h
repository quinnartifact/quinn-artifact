/**
 * early_exit_shm.h
 *
 * POSIX shared memory for QUINN Early Exit (Dynamic Gradient Pruning)
 *
 * Layout (version 2):
 * - Header (20 bytes): magic, version, num_queries, entry_size, topk_k
 * - Float section:     atomic<float>[num_queries]  — k-th distance, init FLT_MAX
 * - TopK IDs section:  uint32_t[num_queries][topk_k] — SPANN topk VIDs, init UINT32_MAX
 *
 * Write ordering guarantee:
 *   SPANN writes IDs (relaxed), then writes float with memory_order_release.
 *   DiskANN reads float with memory_order_acquire; if != FLT_MAX, IDs are safe to read.
 */

#ifndef QUINN_EARLY_EXIT_SHM_H
#define QUINN_EARLY_EXIT_SHM_H

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <atomic>
#include <cfloat>
#include <climits>

// For POSIX shared memory
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace quinn {

// -----------------------------
// Shared Memory Data Structures
// -----------------------------

struct EarlyExitShmHeader {
    uint32_t magic;        // 0x45584954 = 'EXIT'
    uint32_t version;      // 2
    uint32_t num_queries;  // N
    uint32_t entry_size;   // bytes per float entry (sizeof(float) = 4)
    uint32_t topk_k;       // number of topk IDs stored per query (0 = disabled)
};

// Constants
static constexpr uint32_t kEarlyExitMagic   = 0x45584954;  // 'EXIT'
static constexpr uint32_t kEarlyExitVersion = 2;
static constexpr uint32_t kEarlyExitEntrySize = sizeof(float);
static constexpr size_t   kEarlyExitHeaderSize = sizeof(EarlyExitShmHeader); // 20 bytes

// -----------------------------
// Reader/Writer (DiskANN & SPANN)
// -----------------------------

/**
 * EarlyExitShmAccessor
 *
 * Used by both DiskANN (reader) and SPANN (writer).
 * Maps shared memory and exposes typed access.
 */
class EarlyExitShmAccessor {
public:
    explicit EarlyExitShmAccessor(const std::string& shm_name, size_t expected_queries)
        : shm_name_(shm_name), fd_(-1), map_(nullptr), map_size_(0),
          num_queries_(0), topk_k_(0), entries_(nullptr), topk_ids_(nullptr)
    {
        static_assert(sizeof(std::atomic<float>) == sizeof(float),
                      "std::atomic<float> size mismatch");

        if (shm_name_.empty() || shm_name_[0] != '/')
            throw std::invalid_argument("shm_name must start with '/'");

        open_and_validate(expected_queries);
    }

    ~EarlyExitShmAccessor() { cleanup(); }

    // ----------------------------------------------------------------
    // SPANN write API
    // ----------------------------------------------------------------

    // Write kth dist only (no topk IDs)
    void update(size_t qid, float dist) {
        if (qid < num_queries_)
            entries_[qid].store(dist, std::memory_order_release);
    }

    // Write topk IDs then kth dist (IDs visible to DiskANN after dist is acquired)
    void update_topk(size_t qid, const int64_t* vids, uint32_t count, float kth_dist) {
        if (qid >= num_queries_) return;

        if (topk_ids_ != nullptr && topk_k_ > 0) {
            uint32_t* dst = topk_ids_ + qid * topk_k_;
            uint32_t n = std::min(count, topk_k_);
            for (uint32_t i = 0; i < n; i++)
                dst[i] = (vids[i] >= 0) ? static_cast<uint32_t>(vids[i]) : UINT32_MAX;
            // Fill remaining slots with sentinel
            for (uint32_t i = n; i < topk_k_; i++)
                dst[i] = UINT32_MAX;
        }

        // Release fence: IDs are visible to any thread that acquires this store
        entries_[qid].store(kth_dist, std::memory_order_release);
    }

    // ----------------------------------------------------------------
    // DiskANN read API
    // ----------------------------------------------------------------

    // Read kth dist (FLT_MAX means SPANN not done yet)
    float get(size_t qid) const {
        if (qid < num_queries_)
            return entries_[qid].load(std::memory_order_acquire);
        return FLT_MAX;
    }

    // Read topk IDs into caller-provided buffer.
    // Returns number of valid IDs copied (0 if topk disabled or SPANN not done).
    // Must only be called after get(qid) returns != FLT_MAX.
    uint32_t get_topk(size_t qid, uint32_t* ids_out, uint32_t max_count) const {
        if (qid >= num_queries_ || topk_ids_ == nullptr || topk_k_ == 0)
            return 0;
        // Acquire ordering already established by the get() call above.
        // Plain load is sufficient here.
        const uint32_t* src = topk_ids_ + qid * topk_k_;
        uint32_t n = std::min(max_count, topk_k_);
        uint32_t valid = 0;
        for (uint32_t i = 0; i < n; i++) {
            ids_out[i] = src[i];
            if (src[i] != UINT32_MAX) valid++;
        }
        return valid;
    }

    size_t size()   const { return num_queries_; }
    uint32_t topk_k() const { return topk_k_; }

private:
    void open_and_validate(size_t expected_queries) {
        fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0);
        if (fd_ < 0)
            throw std::runtime_error("shm_open failed: " + std::string(strerror(errno)));

        struct stat sb;
        if (fstat(fd_, &sb) != 0) { close(fd_); throw std::runtime_error("fstat failed"); }
        map_size_ = sb.st_size;

        if (map_size_ < kEarlyExitHeaderSize) {
            close(fd_);
            throw std::runtime_error("shm too small");
        }

        map_ = mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (map_ == MAP_FAILED) { close(fd_); throw std::runtime_error("mmap failed"); }

        auto* hdr = reinterpret_cast<EarlyExitShmHeader*>(map_);
        if (hdr->magic != kEarlyExitMagic)
            throw std::runtime_error("EarlyExitShm: bad magic");
        // Accept version 1 (no topk) and version 2
        if (hdr->version != 1 && hdr->version != 2)
            throw std::runtime_error("EarlyExitShm: unsupported version");

        num_queries_ = hdr->num_queries;
        topk_k_      = (hdr->version >= 2) ? hdr->topk_k : 0;

        if (expected_queries > 0 && num_queries_ != expected_queries)
            std::cerr << "Warning: SHM has " << num_queries_
                      << " queries, expected " << expected_queries << std::endl;

        char* base = static_cast<char*>(map_);
        entries_ = reinterpret_cast<std::atomic<float>*>(base + kEarlyExitHeaderSize);

        if (topk_k_ > 0) {
            size_t float_section = num_queries_ * sizeof(float);
            topk_ids_ = reinterpret_cast<uint32_t*>(base + kEarlyExitHeaderSize + float_section);
        }
    }

    void cleanup() {
        if (map_ != nullptr && map_ != MAP_FAILED) { munmap(map_, map_size_); map_ = nullptr; }
        if (fd_ >= 0) { close(fd_); fd_ = -1; }
    }

    std::string          shm_name_;
    int                  fd_;
    void*                map_;
    size_t               map_size_;
    size_t               num_queries_;
    uint32_t             topk_k_;
    std::atomic<float>*  entries_;
    uint32_t*            topk_ids_;  // non-owning pointer into mmap
};

} // namespace quinn

#endif // QUINN_EARLY_EXIT_SHM_H
