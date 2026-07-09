/**
 * budget_shm.h
 *
 * POSIX shared memory for QUINN per-query budgets
 *
 * Follows the shared memory format defined in ./doc/controller.md:
 * - Header: magic, version, num_queries, entry_size
 * - Entries: array of (bS, bD) pairs
 */

#ifndef QUINN_BUDGET_SHM_H
#define QUINN_BUDGET_SHM_H

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

// For POSIX shared memory
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace quinn {

// -----------------------------
// Shared Memory Data Structures
// -----------------------------

/**
 * Shared Memory Header (fixed little-endian)
 *
 * Total size: 16 bytes
 */
struct BudgetShmHeader {
    uint32_t magic;        // 0x43415341 = 'CASA'
    uint32_t version;      // 1
    uint32_t num_queries;  // N
    uint32_t entry_size;   // bytes per entry (fixed at 4 bytes)
};

/**
 * Budget Entry (per query)
 *
 * Total size: 4 bytes
 */
struct BudgetEntry {
    uint16_t bS;  // SPANN nprobe (0..200)
    uint16_t bD;  // DiskANN L (0..200)
};

// Constants
static constexpr uint32_t kMagic = 0x43415341;   // 'CASA'
static constexpr uint32_t kVersion = 1;
static constexpr uint32_t kEntrySize = sizeof(BudgetEntry);

// -----------------------------
// Shared Memory Writer (Controller)
// -----------------------------

/**
 * BudgetShmWriter
 *
 * Used on the controller side, responsible for:
 * 1. Creating the shared memory
 * 2. Writing the budgets
 * 3. Releasing resources
 */
class BudgetShmWriter {
public:
    /**
     * Constructor
     *
     * @param shm_name Shared memory name (must start with '/')
     * @param budgets Budget entries (bS, bD pairs)
     */
    BudgetShmWriter(const std::string& shm_name, const std::vector<BudgetEntry>& budgets)
        : shm_name_(shm_name), budgets_(budgets), fd_(-1), map_(nullptr), map_size_(0)
    {
        if (shm_name_.empty() || shm_name_[0] != '/') {
            throw std::invalid_argument("shm_name must start with '/'");
        }

        if (budgets_.empty()) {
            throw std::invalid_argument("budgets cannot be empty");
        }

        create_and_fill();
    }

    /**
     * Destructor - automatically releases resources
     */
    ~BudgetShmWriter() {
        cleanup();
    }

    /**
     * Get the shared memory name (used to pass to the child process)
     */
    const std::string& name() const {
        return shm_name_;
    }

    /**
     * Get the number of queries
     */
    size_t num_queries() const {
        return budgets_.size();
    }

private:
    void create_and_fill() {
        // Compute the shared memory size
        map_size_ = sizeof(BudgetShmHeader) + budgets_.size() * sizeof(BudgetEntry);

        // Create the shared memory
        fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0600);
        if (fd_ < 0) {
            throw std::runtime_error("shm_open failed: " + std::string(strerror(errno)));
        }

        // Set the size
        if (ftruncate(fd_, map_size_) != 0) {
            close(fd_);
            shm_unlink(shm_name_.c_str());
            throw std::runtime_error("ftruncate failed: " + std::string(strerror(errno)));
        }

        // mmap
        map_ = mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (map_ == MAP_FAILED) {
            close(fd_);
            shm_unlink(shm_name_.c_str());
            throw std::runtime_error("mmap failed: " + std::string(strerror(errno)));
        }

        // Write the header
        auto* hdr = reinterpret_cast<BudgetShmHeader*>(map_);
        hdr->magic = kMagic;
        hdr->version = kVersion;
        hdr->num_queries = static_cast<uint32_t>(budgets_.size());
        hdr->entry_size = kEntrySize;

        // Write the entries
        auto* arr = reinterpret_cast<BudgetEntry*>(static_cast<char*>(map_) + sizeof(BudgetShmHeader));
        std::memcpy(arr, budgets_.data(), budgets_.size() * sizeof(BudgetEntry));

        // Ensure the write completes (optional, usually unnecessary for shm)
        // msync(map_, map_size_, MS_SYNC);

        std::cout << "[BudgetShmWriter] Created shm: " << shm_name_
                  << " (" << budgets_.size() << " queries, " << map_size_ << " bytes)" << std::endl;
    }

    void cleanup() {
        if (map_ != nullptr && map_ != MAP_FAILED) {
            munmap(map_, map_size_);
            map_ = nullptr;
        }

        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }

        // Delete the shared memory (important! avoids garbage piling up in /dev/shm)
        if (!shm_name_.empty()) {
            shm_unlink(shm_name_.c_str());
            std::cout << "[BudgetShmWriter] Cleaned up shm: " << shm_name_ << std::endl;
        }
    }

    std::string shm_name_;
    std::vector<BudgetEntry> budgets_;
    int fd_;
    void* map_;
    size_t map_size_;
};

// -----------------------------
// Shared Memory Reader (DiskANN/SPANN CLI)
// -----------------------------

/**
 * BudgetShmReader
 *
 * Used on the DiskANN/SPANN CLI side, responsible for:
 * 1. Opening the shared memory
 * 2. Validating the header
 * 3. Reading the budgets
 */
class BudgetShmReader {
public:
    /**
     * Constructor
     *
     * @param shm_name Shared memory name
     */
    explicit BudgetShmReader(const std::string& shm_name)
        : shm_name_(shm_name), fd_(-1), map_(nullptr), map_size_(0), num_queries_(0), entries_(nullptr)
    {
        if (shm_name_.empty() || shm_name_[0] != '/') {
            throw std::invalid_argument("shm_name must start with '/'");
        }

        open_and_validate();
    }

    /**
     * Destructor
     */
    ~BudgetShmReader() {
        cleanup();
    }

    /**
     * Get the budget entry for a given qid
     *
     * @param qid Query ID (0-indexed)
     * @return BudgetEntry
     */
    BudgetEntry get(size_t qid) const {
        if (qid >= num_queries_) {
            throw std::out_of_range("qid out of range");
        }
        return entries_[qid];
    }

    /**
     * Get the number of queries
     */
    size_t size() const {
        return num_queries_;
    }

private:
    void open_and_validate() {
        // Open the shared memory
        fd_ = shm_open(shm_name_.c_str(), O_RDONLY, 0);
        if (fd_ < 0) {
            throw std::runtime_error("shm_open failed: " + std::string(strerror(errno)));
        }

        // Get the size
        struct stat sb;
        if (fstat(fd_, &sb) != 0) {
            close(fd_);
            throw std::runtime_error("fstat failed: " + std::string(strerror(errno)));
        }
        map_size_ = sb.st_size;

        // Check the minimum size
        if (map_size_ < sizeof(BudgetShmHeader)) {
            close(fd_);
            throw std::runtime_error("shm too small");
        }

        // mmap
        map_ = mmap(nullptr, map_size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (map_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("mmap failed: " + std::string(strerror(errno)));
        }

        // Validate the header
        auto* hdr = reinterpret_cast<const BudgetShmHeader*>(map_);

        if (hdr->magic != kMagic) {
            cleanup();
            throw std::runtime_error("Invalid magic number");
        }

        if (hdr->version != kVersion) {
            cleanup();
            throw std::runtime_error("Unsupported version");
        }

        if (hdr->entry_size != kEntrySize) {
            cleanup();
            throw std::runtime_error("Entry size mismatch");
        }

        num_queries_ = hdr->num_queries;

        // Check that the size is correct
        size_t expected_size = sizeof(BudgetShmHeader) + num_queries_ * sizeof(BudgetEntry);
        if (map_size_ < expected_size) {
            cleanup();
            throw std::runtime_error("shm size mismatch");
        }

        // Set up the entries pointer
        entries_ = reinterpret_cast<const BudgetEntry*>(static_cast<const char*>(map_) + sizeof(BudgetShmHeader));

        std::cout << "[BudgetShmReader] Opened shm: " << shm_name_
                  << " (" << num_queries_ << " queries)" << std::endl;
    }

    void cleanup() {
        if (map_ != nullptr && map_ != MAP_FAILED) {
            munmap(map_, map_size_);
            map_ = nullptr;
        }

        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

    std::string shm_name_;
    int fd_;
    void* map_;
    size_t map_size_;
    size_t num_queries_;
    const BudgetEntry* entries_;
};

}  // namespace quinn

#endif  // QUINN_BUDGET_SHM_H
