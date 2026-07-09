// io_trace.h — Header-only I/O tracing library for QUINN
// Generates Chrome Trace Event JSON viewable in https://ui.perfetto.dev
//
// Usage:
//   1. #include "io_trace.h"
//   2. Call io_trace::init("process_name") at startup
//   3. Call io_trace::record_io(...) around io_submit/io_getevents
//   4. Call io_trace::flush("output.json") before exit
//
// Thread-safe: each thread collects into its own buffer, flush merges all.

#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace io_trace {

// ─── Types ────────────────────────────────────────────────────
struct IOEvent {
    int64_t  submit_us;    // microseconds since process start
    int64_t  dur_us;       // duration (complete - submit)
    uint64_t tid;          // OS thread id
    uint64_t offset;       // file offset of this read
    uint64_t size;         // bytes read
    std::string name;      // event name (e.g. "io_read")
};

// ─── Global State ─────────────────────────────────────────────
namespace detail {

inline std::mutex& global_mutex() {
    static std::mutex m;
    return m;
}

inline std::vector<IOEvent>& global_events() {
    static std::vector<IOEvent> evts;
    return evts;
}

inline std::string& process_name() {
    static std::string name = "unknown";
    return name;
}

inline bool& enabled() {
    static bool e = false;
    return e;
}

inline std::string& output_path() {
    static std::string path;
    return path;
}

// Epoch: the time point when init() was called
inline std::chrono::steady_clock::time_point& epoch() {
    static auto ep = std::chrono::steady_clock::now();
    return ep;
}

// Thread-local event buffer to avoid lock contention during recording
inline std::vector<IOEvent>& tl_events() {
    thread_local std::vector<IOEvent> evts;
    return evts;
}

inline uint64_t get_tid() {
    // Use a hash of std::thread::id as a stable numeric tid
    static thread_local uint64_t cached_tid = 0;
    if (cached_tid == 0) {
        std::ostringstream oss;
        oss << std::this_thread::get_id();
        cached_tid = std::stoull(oss.str());
    }
    return cached_tid;
}

} // namespace detail

// ─── Public API ───────────────────────────────────────────────

/// Initialize tracing. Call once at startup.
/// @param name Process name (e.g. "diskann" or "spann")
/// @param trace_path Output file path. If empty, tracing is disabled.
inline void init(const std::string& name, const std::string& trace_path) {
    detail::process_name() = name;
    detail::output_path() = trace_path;
    detail::enabled() = !trace_path.empty();
    detail::epoch() = std::chrono::steady_clock::now();
    if (detail::enabled()) {
        fprintf(stderr, "[io_trace] Enabled for '%s', output: %s, enabled_ptr: %p\n",
                name.c_str(), trace_path.c_str(), (void*)&detail::enabled());
    }
}

/// Record a batch of I/O events.
/// Call this after io_getevents returns (i.e., after all reads in the batch complete).
/// @param event_name  Category name (e.g. "io_read")
/// @param submit_time Time point when io_submit was called
/// @param complete_time Time point when io_getevents returned
/// @param offsets     Array of file offsets for each request in the batch
/// @param sizes       Array of read sizes for each request in the batch
/// @param n           Number of requests in this batch
inline void record_batch(
    const std::string& event_name,
    std::chrono::steady_clock::time_point submit_time,
    std::chrono::steady_clock::time_point complete_time,
    const uint64_t* offsets,
    const uint64_t* sizes,
    uint64_t n)
{
    if (!detail::enabled()) return;

    auto& ep = detail::epoch();
    int64_t submit_us = std::chrono::duration_cast<std::chrono::microseconds>(
        submit_time - ep).count();
    int64_t dur_us = std::chrono::duration_cast<std::chrono::microseconds>(
        complete_time - submit_time).count();
    uint64_t tid = detail::get_tid();

    auto& tl = detail::tl_events();
    static bool first_record = true;
    if (first_record) {
        fprintf(stderr, "[io_trace] First record in process '%s', enabled: %d, events_ptr: %p\n",
                detail::process_name().c_str(), detail::enabled(), (void*)&detail::enabled());
        first_record = false;
    }
    for (uint64_t i = 0; i < n; i++) {
        tl.push_back(IOEvent{
            submit_us, dur_us, tid,
            offsets[i], sizes[i], event_name
        });
    }
}

/// Convenience: record a single I/O event
inline void record_single(
    const std::string& event_name,
    std::chrono::steady_clock::time_point submit_time,
    std::chrono::steady_clock::time_point complete_time,
    uint64_t offset,
    uint64_t size)
{
    record_batch(event_name, submit_time, complete_time, &offset, &size, 1);
}

/// Flush thread-local events to global buffer.
/// Call this from each worker thread before it exits.
inline void flush_thread() {
    if (!detail::enabled()) return;

    auto& tl = detail::tl_events();
    if (tl.empty()) return;

    std::lock_guard<std::mutex> lk(detail::global_mutex());
    auto& g = detail::global_events();
    g.insert(g.end(), tl.begin(), tl.end());
    tl.clear();
}

/// Write all collected events to a Chrome Trace Event JSON file.
/// Call once after all threads have completed and called flush_thread().
inline void flush(const std::string& override_path = "") {
    if (!detail::enabled()) return;

    // Also drain the calling thread's buffer
    flush_thread();

    std::string path = override_path.empty() ? detail::output_path() : override_path;
    if (path.empty()) return;

    std::lock_guard<std::mutex> lk(detail::global_mutex());
    auto& events = detail::global_events();

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        fprintf(stderr, "[io_trace] ERROR: Cannot open %s for writing\n", path.c_str());
        return;
    }

    ofs << "{\"traceEvents\":[\n";

    // Write a process name metadata event
    ofs << "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"tid\":0,"
        << "\"args\":{\"name\":\"" << detail::process_name() << "\"}}\n";

    for (size_t i = 0; i < events.size(); i++) {
        auto& e = events[i];
        ofs << ",{\"name\":\"" << e.name << "\""
            << ",\"cat\":\"io\""
            << ",\"ph\":\"X\""
            << ",\"ts\":" << e.submit_us
            << ",\"dur\":" << e.dur_us
            << ",\"pid\":1"
            << ",\"tid\":" << e.tid
            << ",\"args\":{"
            << "\"offset\":" << e.offset
            << ",\"size\":" << e.size
            << "}}\n";
    }

    ofs << "]}\n";
    ofs.close();

    fprintf(stderr, "[io_trace] Wrote %zu events to %s\n",
            events.size(), path.c_str());
}

} // namespace io_trace
