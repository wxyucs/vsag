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
    bool use_conjugate_graph;
    bool use_static;

protected:
    CreateHnswParameters() = default;
};

struct CreateFreshHnswParameters : public CreateHnswParameters {
public:
    static CreateFreshHnswParameters
    FromJson(const std::string& json_string);

public:
    // required vars
    bool use_reversed_edges;

private:
    CreateFreshHnswParameters() = default;
};

struct HnswSearchParameters {
public:
    static HnswSearchParameters
    FromJson(const std::string& json_string);

public:
    // required vars
    int64_t ef_search;
    bool use_conjugate_graph_search;

private:
    HnswSearchParameters() = default;
};

}  // namespace vsag
