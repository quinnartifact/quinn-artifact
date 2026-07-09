// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "linux_aligned_file_reader.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include "tsl/robin_map.h"
#include "utils.h"
#include "io_trace.h"
#define MAX_EVENTS 1024

namespace
{
typedef struct io_event io_event_t;
typedef struct iocb iocb_t;

void execute_io(io_context_t ctx, int fd, std::vector<AlignedRead> &read_reqs, uint64_t n_retries = 0)
{
#ifdef DEBUG
    for (auto &req : read_reqs)
    {
        assert(IS_ALIGNED(req.len, 512));
        // std::cout << "request:"<<req.offset<<":"<<req.len << std::endl;
        assert(IS_ALIGNED(req.offset, 512));
        assert(IS_ALIGNED(req.buf, 512));
        // assert(malloc_usable_size(req.buf) >= req.len);
    }
#endif

    // break-up requests into chunks of size MAX_EVENTS each
    uint64_t n_iters = ROUND_UP(read_reqs.size(), MAX_EVENTS) / MAX_EVENTS;
    for (uint64_t iter = 0; iter < n_iters; iter++)
    {
        uint64_t n_ops = std::min((uint64_t)read_reqs.size() - (iter * MAX_EVENTS), (uint64_t)MAX_EVENTS);
        std::vector<iocb_t *> cbs(n_ops, nullptr);
        std::vector<io_event_t> evts(n_ops);
        std::vector<struct iocb> cb(n_ops);
        for (uint64_t j = 0; j < n_ops; j++)
        {
            io_prep_pread(cb.data() + j, fd, read_reqs[j + iter * MAX_EVENTS].buf, read_reqs[j + iter * MAX_EVENTS].len,
                          read_reqs[j + iter * MAX_EVENTS].offset);
        }

        // initialize `cbs` using `cb` array
        //

        for (uint64_t i = 0; i < n_ops; i++)
        {
            cbs[i] = cb.data() + i;
        }

        uint64_t n_tries = 0;
        while (n_tries <= n_retries)
        {
            // io_trace: capture submit timestamp
            auto _trace_submit_ts = std::chrono::steady_clock::now();

            // issue reads
            int64_t ret = io_submit(ctx, (int64_t)n_ops, cbs.data());
            // if requests didn't get accepted
            if (ret != (int64_t)n_ops)
            {
                std::cerr << "io_submit() failed; returned " << ret << ", expected=" << n_ops << ", ernno=" << errno
                          << "=" << ::strerror(-ret) << ", try #" << n_tries + 1;
                std::cout << "ctx: " << ctx << "\n";
                exit(-1);
            }
            else
            {
                // wait on io_getevents
                ret = io_getevents(ctx, (int64_t)n_ops, (int64_t)n_ops, evts.data(), nullptr);
                // if requests didn't complete
                if (ret != (int64_t)n_ops)
                {
                    std::cerr << "io_getevents() failed; returned " << ret << ", expected=" << n_ops
                              << ", ernno=" << errno << "=" << ::strerror(-ret) << ", try #" << n_tries + 1;
                    exit(-1);
                }
                else
                {
                    // io_trace: capture complete timestamp and record events
                    auto _trace_complete_ts = std::chrono::steady_clock::now();
                    {
                        uint64_t base = iter * MAX_EVENTS;
                        std::vector<uint64_t> _t_offsets(n_ops), _t_sizes(n_ops);
                        for (uint64_t j = 0; j < n_ops; j++) {
                            _t_offsets[j] = read_reqs[j + base].offset;
                            _t_sizes[j]   = read_reqs[j + base].len;
                        }
                        io_trace::record_batch("io_read",
                            _trace_submit_ts, _trace_complete_ts,
                            _t_offsets.data(), _t_sizes.data(), n_ops);
                    }
                    break;
                }
            }
        }
        // disabled since req.buf could be an offset into another buf
        /*
        for (auto &req : read_reqs) {
          // corruption check
          assert(malloc_usable_size(req.buf) >= req.len);
        }
        */
    }
}
} // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader()
{
    this->file_desc = -1;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader()
{
    int64_t ret;
    // check to make sure file_desc is closed
    ret = ::fcntl(this->file_desc, F_GETFD);
    if (ret == -1)
    {
        if (errno != EBADF)
        {
            std::cerr << "close() not called" << std::endl;
            // close file desc
            ret = ::close(this->file_desc);
            // error checks
            if (ret == -1)
            {
                std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno)
                          << std::endl;
            }
        }
    }
}

io_context_t &LinuxAlignedFileReader::get_ctx()
{
    std::unique_lock<std::mutex> lk(ctx_mut);
    // perform checks only in DEBUG mode
    if (ctx_map.find(std::this_thread::get_id()) == ctx_map.end())
    {
        std::cerr << "bad thread access; returning -1 as io_context_t" << std::endl;
        return this->bad_ctx;
    }
    else
    {
        return ctx_map[std::this_thread::get_id()];
    }
}

void LinuxAlignedFileReader::register_thread()
{
    auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut);
    if (ctx_map.find(my_id) != ctx_map.end())
    {
        std::cerr << "multiple calls to register_thread from the same thread" << std::endl;
        return;
    }
    io_context_t ctx = 0;
    int ret = io_setup(MAX_EVENTS, &ctx);
    if (ret != 0)
    {
        lk.unlock();
        if (ret == -EAGAIN)
        {
            std::cerr << "io_setup() failed with EAGAIN: Consider increasing /proc/sys/fs/aio-max-nr" << std::endl;
        }
        else
        {
            std::cerr << "io_setup() failed; returned " << ret << ": " << ::strerror(-ret) << std::endl;
        }
    }
    else
    {
        diskann::cout << "allocating ctx: " << ctx << " to thread-id:" << my_id << std::endl;
        ctx_map[my_id] = ctx;
    }

    // QUINN: also allocate a separate prefetch io_context for this thread
    io_context_t pctx = 0;
    if (io_setup(MAX_EVENTS, &pctx) == 0)
    {
        prefetch_state_map[my_id].ctx = pctx;
    }

    lk.unlock();
}

void LinuxAlignedFileReader::deregister_thread()
{
    auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut);
    assert(ctx_map.find(my_id) != ctx_map.end());

    lk.unlock();
    io_context_t ctx = this->get_ctx();
    io_destroy(ctx);
    //  assert(ret == 0);
    lk.lock();
    ctx_map.erase(my_id);
    std::cerr << "returned ctx from thread-id:" << my_id << std::endl;

    // QUINN: also destroy prefetch context
    auto pit = prefetch_state_map.find(my_id);
    if (pit != prefetch_state_map.end())
    {
        if (pit->second.ctx != 0)
            io_destroy(pit->second.ctx);
        prefetch_state_map.erase(pit);
    }

    lk.unlock();
}

void LinuxAlignedFileReader::deregister_all_threads()
{
    std::unique_lock<std::mutex> lk(ctx_mut);
    for (auto x = ctx_map.begin(); x != ctx_map.end(); x++)
    {
        io_context_t ctx = x.value();
        io_destroy(ctx);
    }
    ctx_map.clear();

    // QUINN: also destroy all prefetch contexts
    for (auto x = prefetch_state_map.begin(); x != prefetch_state_map.end(); x++)
    {
        if (x->second.ctx != 0)
            io_destroy(x->second.ctx);
    }
    prefetch_state_map.clear();
}

void LinuxAlignedFileReader::open(const std::string &fname)
{
    int flags = O_DIRECT | O_RDONLY | O_LARGEFILE;
    this->file_desc = ::open(fname.c_str(), flags);
    // error checks
    assert(this->file_desc != -1);
    std::cerr << "Opened file : " << fname << std::endl;
}

void LinuxAlignedFileReader::close()
{
    //  int64_t ret;

    // check to make sure file_desc is closed
    ::fcntl(this->file_desc, F_GETFD);
    //  assert(ret != -1);

    ::close(this->file_desc);
    //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<AlignedRead> &read_reqs, io_context_t &ctx, bool async)
{
    if (async == true)
    {
        diskann::cout << "Async currently not supported in linux." << std::endl;
    }
    assert(this->file_desc != -1);
    execute_io(ctx, this->file_desc, read_reqs);
}

// QUINN: submit pre-fetch IOs without waiting (non-blocking).
// Destination buffers in read_reqs must stay alive until wait_prefetch() returns.
void LinuxAlignedFileReader::submit_prefetch(std::vector<AlignedRead> &read_reqs)
{
    if (read_reqs.empty())
        return;

    auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut);
    auto pit = prefetch_state_map.find(my_id);
    if (pit == prefetch_state_map.end() || pit.value().ctx == 0)
        return;
    PrefetchState &ps = pit.value();
    lk.unlock();

    int64_t n_ops = static_cast<int64_t>(read_reqs.size());
    ps.cbs.resize(n_ops);
    std::vector<struct iocb *> cbptrs(n_ops);
    for (int64_t j = 0; j < n_ops; j++)
    {
        io_prep_pread(&ps.cbs[j], this->file_desc,
                      read_reqs[j].buf, read_reqs[j].len, read_reqs[j].offset);
        cbptrs[j] = &ps.cbs[j];
    }

    int64_t ret = io_submit(ps.ctx, n_ops, cbptrs.data());
    ps.n_pending = (ret > 0) ? ret : 0;
}

// QUINN: wait for all IOs submitted by the last submit_prefetch() call.
void LinuxAlignedFileReader::wait_prefetch()
{
    auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut);
    auto pit = prefetch_state_map.find(my_id);
    if (pit == prefetch_state_map.end() || pit.value().n_pending == 0)
        return;
    PrefetchState &ps = pit.value();
    lk.unlock();

    ps.evts.resize(ps.n_pending);
    io_getevents(ps.ctx, ps.n_pending, ps.n_pending, ps.evts.data(), nullptr);
    ps.n_pending = 0;
}
