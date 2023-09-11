// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#ifndef _WINDOWS

#include "aligned_file_reader.h"

class LinuxAlignedFileReader : public AlignedFileReader
{
  private:
    uint64_t file_sz;
    FileHandle file_desc;
    io_context_t bad_ctx = (io_context_t)-1;

  public:
    LinuxAlignedFileReader();
    ~LinuxAlignedFileReader();

    IOContext &get_ctx() override;

    // register thread-id for a context
    void register_thread() override;

    // de-register thread-id for a context
    void deregister_thread() override;
    void deregister_all_threads() override;

    // Open & close ops
    // Blocking calls
    void open(const std::string &fname) override;
    void close() override;

    // process batch of aligned requests in parallel
    // NOTE :: blocking call
    void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool async = false) override;
};

#endif
