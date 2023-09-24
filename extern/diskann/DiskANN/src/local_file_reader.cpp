//
// Created by root on 9/22/23.
//


#include "local_file_reader.h"


void LocalFileReader::read(std::vector <AlignedRead> &read_reqs, IOContext &ctx, bool async) {
    batch_request batch;
    for (int i = 0; i < read_reqs.size(); ++i) {
        batch.emplace_back(read_reqs[i].offset, read_reqs[i].len, read_reqs[i].buf);
    }
    func_(batch);
}