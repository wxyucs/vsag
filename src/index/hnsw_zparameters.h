#pragma once

#include <hnswlib/hnswlib.h>

#include <memory>
#include <string>

namespace vsag {

struct CreateHnswParameters {
public:
    static CreateHnswParameters
    FromJson(const std::string& json_string);

public:
    // required vars
    std::shared_ptr<hnswlib::SpaceInterface> space;
    int64_t max_degree;
    int64_t ef_construction;
    bool use_static;
    bool use_reversed_edges;

private:
    CreateHnswParameters() = default;
};

struct HnswSearchParameters {
public:
    static HnswSearchParameters
    FromJson(const std::string& json_string);

public:
    // required vars
    int64_t ef_search;

private:
    HnswSearchParameters() = default;
};

}  // namespace vsag
