#pragma once
#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "vsag/constants.h"

namespace vsag {

static const std::string MAGIC_NUM = "43475048";  // means "CGPH"
static const std::string VERSION = "1";
static const int FOOTER_SIZE = 4096;  // 4KB

class SerializationFooter {
public:
    SerializationFooter();

    void
    Clear();

    void
    SetMetadata(const std::string& key, const std::string& value);

    std::string
    GetMetadata(const std::string& key) const;

    void
    Serialize(std::ostream& out_stream) const;

    void
    Deserialize(std::istream& in_stream);

private:
    nlohmann::json json_;
};

}  // namespace vsag
