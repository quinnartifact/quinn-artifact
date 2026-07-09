#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace quinn {

struct ThreadCountData {
    uint32_t magic;
    uint32_t version;
    int32_t  thread_s;
    int32_t  thread_d;
};
static_assert(sizeof(ThreadCountData) == 16, "ThreadCountData must be 16 bytes");

class ThreadCountShmReader {
public:
    static constexpr uint32_t MAGIC   = 0x54485243;
    static constexpr uint32_t VERSION = 1;

    explicit ThreadCountShmReader(const std::string& name) {
        int fd = ::shm_open(name.c_str(), O_RDONLY, 0666);
        if (fd < 0) throw std::runtime_error("ThreadCountShmReader: shm_open failed: " + name);
        _data = static_cast<const ThreadCountData*>(
            ::mmap(nullptr, sizeof(ThreadCountData), PROT_READ, MAP_SHARED, fd, 0));
        ::close(fd);
        if (_data == MAP_FAILED) throw std::runtime_error("ThreadCountShmReader: mmap failed");
    }

    ~ThreadCountShmReader() {
        if (_data && _data != MAP_FAILED)
            ::munmap(const_cast<ThreadCountData*>(_data), sizeof(ThreadCountData));
    }

    int32_t get_thread_s() const { return __atomic_load_n(&_data->thread_s, __ATOMIC_ACQUIRE); }
    int32_t get_thread_d() const { return __atomic_load_n(&_data->thread_d, __ATOMIC_ACQUIRE); }

private:
    const ThreadCountData* _data = nullptr;
};

} // namespace quinn
