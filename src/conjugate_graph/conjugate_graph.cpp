#include "conjugate_graph.h"

namespace vsag {

const static int ENHANCE_ROUND = 2;

ConjugateGraph::ConjugateGraph() {
    memory_usage_ = sizeof(memory_usage_);
}

tl::expected<bool, Error>
ConjugateGraph::AddNeighbor(int64_t from_tag_id, int64_t to_tag_id) {
    auto& neighbor_set = conjugate_graph_[from_tag_id];
    auto insert_result = neighbor_set.insert(to_tag_id);
    if (!insert_result.second) {
        return false;
    } else {
        if (neighbor_set.size() == 1) {
            memory_usage_ += sizeof(from_tag_id);
            memory_usage_ += sizeof(neighbor_set.size());
        }
        memory_usage_ += sizeof(to_tag_id);
        return true;
    }
}

const std::unordered_set<int64_t>&
ConjugateGraph::get_neighbors(int64_t from_tag_id) const {
    static const std::unordered_set<int64_t> empty_set;
    auto search = conjugate_graph_.find(from_tag_id);
    if (search != conjugate_graph_.end()) {
        return search->second;
    } else {
        return empty_set;
    }
}

tl::expected<uint32_t, Error>
ConjugateGraph::EnhanceResult(std::priority_queue<std::pair<float, size_t>>& results,
                              const std::function<float(int64_t)>& distance_of_tag) const {
    int64_t k = results.size();
    bool find_new_local_optimum = false;
    std::priority_queue<std::pair<float, size_t>> old_results(results);
    std::unordered_set<int64_t> results_set;
    int64_t local_optimum_tag_id;
    float local_optimum_dist;
    uint32_t successfully_enhanced = 0;
    float distance = 0;

    // find current local optimum
    for (int64_t j = old_results.size() - 1; j >= 0; --j) {
        local_optimum_dist = old_results.top().first;
        local_optimum_tag_id = old_results.top().second;
        results_set.insert(local_optimum_tag_id);
        old_results.pop();
    }

    // multi-rounds enhancement for routing to global optimum
    for (int i = 0; i < ENHANCE_ROUND; i++) {
        const std::unordered_set<int64_t>& neighbors = get_neighbors(local_optimum_tag_id);

        for (auto neighbor_tag_id : neighbors) {
            if (results_set.find(neighbor_tag_id) != results_set.end()) {
                continue;
            }
            distance = distance_of_tag(neighbor_tag_id);

            // insert into results
            if (distance < results.top().first) {
                results.emplace(distance, neighbor_tag_id);
                results_set.insert(neighbor_tag_id);
            }

            // update current local optimum
            if (distance < local_optimum_dist) {
                find_new_local_optimum = true;
                local_optimum_tag_id = neighbor_tag_id;
                local_optimum_dist = distance;
            }
        }

        if (not find_new_local_optimum) {
            break;
        }
    }

    // result clean up
    for (auto j = results.size(); j > k; j--) {
        results.pop();
        successfully_enhanced++;
    }
    return successfully_enhanced;
}

size_t
ConjugateGraph::GetMemoryUsage() const {
    return memory_usage_;
}

}  // namespace vsag