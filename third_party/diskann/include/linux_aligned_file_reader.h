// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#ifndef _WINDOWS

#include <vector>
#include <libaio.h>
#include "aligned_file_reader.h"

// Per-thread state for non-blocking (async) pre-fetch IOs.
// iocb structs must remain alive between io_submit and io_getevents.
struct PrefetchState
{
    io_context_t      ctx       = 0;
    std::vector<struct iocb>     cbs;     // kept alive until wait_prefetch()
    std::vector<struct io_event> evts;
    int64_t           n_pending = 0;
};

class LinuxAlignedFileReader : public AlignedFileReader
{
  private:
    uint64_t file_sz;
    FileHandle file_desc;
    io_context_t bad_ctx = (io_context_t)-1;

    // Per-thread prefetch contexts (separate from main ctx to avoid event interleaving)
    tsl::robin_map<std::thread::id, PrefetchState> prefetch_state_map;

  public:
    LinuxAlignedFileReader();
    ~LinuxAlignedFileReader();

    IOContext &get_ctx();

    // register thread-id for a context
    void register_thread();

    // de-register thread-id for a context
    void deregister_thread();
    void deregister_all_threads();

    // Open & close ops
    // Blocking calls
    void open(const std::string &fname);
    void close();

    // process batch of aligned requests in parallel
    // NOTE :: blocking call
    void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool async = false);

    // QUINN: non-blocking submit — returns immediately after io_submit().
    // Caller must call wait_prefetch() before accessing the destination buffers.
    void submit_prefetch(std::vector<AlignedRead> &read_reqs);

    // QUINN: wait for all IOs submitted by the last submit_prefetch() call.
    void wait_prefetch();
};

#endif
