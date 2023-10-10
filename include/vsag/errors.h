#pragma once

namespace vsag {

enum class index_error {
    internal_error,
    build_twice,
    dimension_not_equal,
    no_enough_memory,
    index_not_empty,
    invalid_binary,
};

}  //namespace vsag
