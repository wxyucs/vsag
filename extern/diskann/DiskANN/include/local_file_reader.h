//
// Created by root on 9/22/23.
//

#pragma once

#ifndef _WINDOWS

#include "aligned_file_reader.h"
#include <functional>

typedef std::vector<std::tuple<uint64_t, uint64_t, void*>> batch_request;
typedef std::function<void(batch_request)> reader_function;


class LocalFileReader : public AlignedFileReader
{
private:
    reader_function func_;
    io_context_t bad_ctx;

public:
    LocalFileReader(reader_function func): func_(func) {}
    ~LocalFileReader() {}
    // Open & close ops
    // Blocking calls

    IOContext &get_ctx() override {
        bad_ctx = (io_context_t)-1;
        return bad_ctx;
    }

    // register thread-id for a context
    void register_thread() override {}

    // de-register thread-id for a context
    void deregister_thread() override {}
    void deregister_all_threads() override {}

    // Open & close ops
    // Blocking calls
    void open(const std::string &fname) override {}
    void close() override {}

    // process batch of aligned requests in parallel
    // NOTE :: blocking call
    void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool async = false) override;
};



#endif
