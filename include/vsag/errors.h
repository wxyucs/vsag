#pragma once

#include <sstream>
#include <string>

namespace vsag {

enum class index_error_type {
    internal_error,
    build_twice,
    dimension_not_equal,
    no_enough_memory,
    index_not_empty,
    invalid_binary,
    read_error,
    no_enough_data,
    index_empty,
    missing_file,
    invalid_parameter,
    invalid_index,
    unexpected_error
};

struct index_error {
    index_error(index_error_type t, const std::string& msg) : type(t), message(msg) {
    }

    index_error_type type;
    std::string message;
};

template <typename T>
void
_concate(std::stringstream& ss, const T& value) {
    ss << value;
}

template <typename T, typename... Args>
void
_concate(std::stringstream& ss, const T& value, const Args&... args) {
    ss << value;
    _concate(ss, args...);
}

#define LOG_ERROR_AND_RETURNS(t, ...) \
    std::stringstream ss;             \
    _concate(ss, __VA_ARGS__);        \
    spdlog::error(ss.str());          \
    return tl::unexpected(index_error(t, ss.str()));

}  //namespace vsag
