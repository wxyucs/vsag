#pragma once

#include <nlohmann/json.hpp>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "../logger.h"
#include "vsag/index.h"

namespace vsag {

static const int ENHANCE_ROUND = 2;
static const uint32_t MAGIC_NUM = 0x43475048;  // means "CGPH"
static const uint32_t VERSION = 1;
static const int FOOTER_SIZE = 4096;  // 4KB

class SerializationFooter {
    char json_data[FOOTER_SIZE];

public:
    SerializationFooter() {
        std::memset(json_data, 0, FOOTER_SIZE);
    }

    template <typename T>
    bool
    set_value(const std::string& key, const T& value) {
        nlohmann::json json =
            json_data[0] ? nlohmann::json::parse(json_data) : nlohmann::json::object();

        json[key] = value;

        std::string new_json_str = json.dump();
        if (new_json_str.size() >= FOOTER_SIZE) {
            throw std::runtime_error("Serialized footer size exceeds 4KB");
        }

        std::memset(json_data, 0, FOOTER_SIZE);
        std::memcpy(json_data, new_json_str.c_str(), new_json_str.size());
        return true;
    }

    std::string
    get_value(const std::string& key) const {
        try {
            nlohmann::json json = nlohmann::json::parse(json_data);
            if (json.contains(key)) {
                return json[key].dump();
            } else {
                throw std::runtime_error(fmt::format("Footer doesn't contain key ({})", key));
            }
        } catch (const nlohmann::json::parse_error& e) {
            throw std::runtime_error("Footer json parse error");
        }
    }
};

class ConjugateGraph {
public:
    ConjugateGraph();

    tl::expected<bool, Error>
    AddNeighbor(int64_t from_tag_id, int64_t to_tag_id);

    tl::expected<uint32_t, Error>
    EnhanceResult(std::priority_queue<std::pair<float, size_t>>& results,
                  const std::function<float(int64_t)>& distance_of_tag) const;

public:
    tl::expected<Binary, Error>
    Serialize() const;

    tl::expected<void, Error>
    Serialize(std::ostream& out_stream) const;

    tl::expected<void, Error>
    Deserialize(const Binary& binary);

    tl::expected<void, Error>
    Deserialize(std::istream& in_stream);

    size_t
    GetMemoryUsage() const;

private:
    const std::unordered_set<int64_t>&
    get_neighbors(int64_t from_tag_id) const;

    void
    clean();

private:
    uint32_t memory_usage_;

    std::unordered_map<int64_t, std::unordered_set<int64_t>> conjugate_graph_;

    SerializationFooter footer_;
};

}  // namespace vsag
