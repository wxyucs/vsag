#pragma once

#include <sstream>
#include <string>

namespace vsag {

enum class ErrorType {
    // some internalError in algorithm
    INTERNAL_ERROR,
    // index has been build, cannot build again
    BUILD_TWICE,
    // the dimension of add/build/search request is NOT equal to index
    DIMENSION_NOT_EQUAL,
    // failed to alloc memory
    NO_ENOUGH_MEMORY,
    // index object is NOT empty so that should not deserialize on it
    INDEX_NOT_EMPTY,
    // the content of binary is invalid
    INVALID_BINARY,
    // cannot read from binary
    READ_ERROR,
    // index is empty, cannot search or serialize
    INDEX_EMPTY,
    // some file missing in index diskann deserialization
    MISSING_FILE,
    // invalid argument
    INVALID_ARGUMENT,
    // the index to create is unsupported
    UNSUPPORTED_INDEX,
    // unknown error
    UNKNOWN_ERROR
};

struct Error {
    Error(ErrorType t, const std::string& msg) : type(t), message(msg) {
    }

    ErrorType type;
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
    return tl::unexpected(Error(t, ss.str()));

}  //namespace vsag
