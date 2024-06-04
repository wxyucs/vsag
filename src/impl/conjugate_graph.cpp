#include "conjugate_graph.h"

namespace vsag {

ConjugateGraph::ConjugateGraph() {
    clean();
}

tl::expected<bool, Error>
ConjugateGraph::AddNeighbor(int64_t from_tag_id, int64_t to_tag_id) {
    if (from_tag_id == to_tag_id) {
        return false;
    }
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
    auto iter = conjugate_graph_.find(from_tag_id);
    if (iter != conjugate_graph_.end()) {
        return iter->second;
    } else {
        return empty_set;
    }
}

tl::expected<uint32_t, Error>
ConjugateGraph::EnhanceResult(std::priority_queue<std::pair<float, size_t>>& results,
                              const std::function<float(int64_t)>& distance_of_tag) const {
    int64_t k = results.size();
    int64_t look_at_k = std::min(LOOK_AT_K, k);
    std::priority_queue<std::pair<float, size_t>> old_results(results);
    std::vector<int64_t> to_be_visited(look_at_k);
    std::unordered_set<int64_t> visited_set;
    uint32_t successfully_enhanced = 0;
    float distance = 0;

    // initialize visited_set
    for (int j = old_results.size() - 1; j >= 0; j--) {
        visited_set.insert(old_results.top().second);
        if (j < look_at_k) {
            to_be_visited[j] = old_results.top().second;
        }
        old_results.pop();
    }

    // add neighbors in conjugate graph to enhance result
    for (int j = 0; j < look_at_k; j++) {
        const std::unordered_set<int64_t>& neighbors = get_neighbors(to_be_visited[j]);

        for (auto neighbor_tag_id : neighbors) {
            if (not visited_set.insert(neighbor_tag_id).second) {
                continue;
            }
            distance = distance_of_tag(neighbor_tag_id);
            // insert into results
            if (distance < results.top().first) {
                results.emplace(distance, neighbor_tag_id);
                results.pop();
                successfully_enhanced++;
            }
        }
    }

    return successfully_enhanced;
}

size_t
ConjugateGraph::GetMemoryUsage() const {
    return memory_usage_;
}

template <typename T>
static void
write_var_to_mem(char** dest, const T& ref) {
    std::memcpy(*dest, (char*)&ref, sizeof(T));
    *dest += sizeof(T);
}

template <typename T>
static void
read_var_from_bin(const Binary& binary, uint32_t* offset, T* dest) {
    if (*offset + sizeof(T) > binary.size) {
        throw std::out_of_range("Offset is out of bounds for binary data.");
    }

    std::memcpy(reinterpret_cast<char*>(dest), binary.data.get() + *offset, sizeof(T));
    *offset += sizeof(T);
}

template <typename T>
static void
read_var_from_stream(std::istream& in_stream, uint32_t* offset, T* dest) {
    in_stream.read((char*)dest, sizeof(T));
    *offset += sizeof(T);

    if (in_stream.fail()) {
        throw std::runtime_error("Failed to read from stream.");
    }
}

void
ConjugateGraph::clean() {
    memory_usage_ = sizeof(memory_usage_) + sizeof(footer_);
    footer_.set_value("MAGIC_NUM", MAGIC_NUM);
    footer_.set_value("VERSION", VERSION);
}

tl::expected<Binary, Error>
ConjugateGraph::Serialize() const {
    std::shared_ptr<int8_t[]> bin(new int8_t[memory_usage_]);

    char* dest = reinterpret_cast<char*>(bin.get());
    write_var_to_mem(&dest, memory_usage_);
    for (auto item : conjugate_graph_) {
        auto neighbor_set = item.second;
        write_var_to_mem(&dest, item.first);
        write_var_to_mem(&dest, neighbor_set.size());

        for (auto neighbor_tag_id : neighbor_set) {
            write_var_to_mem(&dest, neighbor_tag_id);
        }
    }
    write_var_to_mem(&dest, footer_);

    Binary binary{.data = bin, .size = memory_usage_};
    return binary;
}

tl::expected<void, Error>
ConjugateGraph::Serialize(std::ostream& out_stream) const {
    out_stream.write((char*)&memory_usage_, sizeof(memory_usage_));

    for (auto item : conjugate_graph_) {
        auto neighbor_set = item.second;
        size_t neighbor_set_size = neighbor_set.size();

        out_stream.write((char*)&item.first, sizeof(item.first));
        out_stream.write((char*)&neighbor_set_size, sizeof(neighbor_set_size));

        for (auto neighbor_tag_id : neighbor_set) {
            out_stream.write((char*)&neighbor_tag_id, sizeof(neighbor_tag_id));
        }
    }

    out_stream.write((char*)&footer_, sizeof(footer_));

    return {};
}

tl::expected<void, Error>
ConjugateGraph::Deserialize(const Binary& binary) {
    try {
        uint32_t offset = 0;
        uint32_t footer_offset = 0;
        size_t neighbor_size = 0;
        int64_t from_tag_id = 0;
        int64_t to_tag_id = 0;

        conjugate_graph_.clear();

        read_var_from_bin(binary, &offset, &memory_usage_);
        if (memory_usage_ <= FOOTER_SIZE) {
            throw std::runtime_error(
                fmt::format("Incorrect header: memory_usage_({})", memory_usage_));
        }
        footer_offset = memory_usage_ - FOOTER_SIZE;

        read_var_from_bin(binary, &footer_offset, &footer_);
        if (std::stoul(footer_.get_value("MAGIC_NUM")) != MAGIC_NUM or
            std::stoul(footer_.get_value("VERSION")) != VERSION) {
            throw std::runtime_error("Incorrect footer");
        }

        offset = sizeof(memory_usage_);
        while (offset != memory_usage_ - FOOTER_SIZE) {
            read_var_from_bin(binary, &offset, &from_tag_id);
            read_var_from_bin(binary, &offset, &neighbor_size);
            for (int i = 0; i < neighbor_size; i++) {
                read_var_from_bin(binary, &offset, &to_tag_id);
                conjugate_graph_[from_tag_id].insert(to_tag_id);
            }
        }

        return {};
    } catch (const std::out_of_range& e) {
        clean();
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    } catch (const std::runtime_error& e) {
        clean();
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    }
}

tl::expected<void, Error>
ConjugateGraph::Deserialize(std::istream& in_stream) {
    try {
        uint32_t offset = 0;
        uint32_t footer_offset = 0;
        size_t neighbor_size = 0;
        int64_t from_tag_id = 0;
        int64_t to_tag_id = 0;

        conjugate_graph_.clear();

        auto cur_pos = in_stream.tellg();

        read_var_from_stream(in_stream, &offset, &memory_usage_);
        if (memory_usage_ <= FOOTER_SIZE) {
            throw std::runtime_error(
                fmt::format("Incorrect header: memory_usage_({})", memory_usage_));
        }
        footer_offset = memory_usage_ - FOOTER_SIZE;

        in_stream.seekg(cur_pos);
        in_stream.seekg(footer_offset, std::ios::cur);
        read_var_from_stream(in_stream, &footer_offset, &footer_);
        if (std::stoul(footer_.get_value("MAGIC_NUM")) != MAGIC_NUM or
            std::stoul(footer_.get_value("VERSION")) != VERSION) {
            throw std::runtime_error("Incorrect footer");
        }

        offset = sizeof(memory_usage_);
        in_stream.seekg(cur_pos);
        in_stream.seekg(offset, std::ios::cur);
        while (offset != memory_usage_ - FOOTER_SIZE) {
            read_var_from_stream(in_stream, &offset, &from_tag_id);
            read_var_from_stream(in_stream, &offset, &neighbor_size);
            for (int i = 0; i < neighbor_size; i++) {
                read_var_from_stream(in_stream, &offset, &to_tag_id);
                conjugate_graph_[from_tag_id].insert(to_tag_id);
            }
        }

        return {};
    } catch (const std::runtime_error& e) {
        clean();
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    }
}

}  // namespace vsag