#pragma once

#define SAFE_CALL(stmt)                                                              \
    try {                                                                            \
        stmt;                                                                        \
    } catch (const std::exception& e) {                                              \
        LOG_ERROR_AND_RETURNS(ErrorType::UNKNOWN_ERROR, "unknownError: ", e.what()); \
    }

#define CHECK_ARGUMENT(expr, message)             \
    do {                                          \
        if (not(expr)) {                          \
            throw std::invalid_argument(message); \
        }                                         \
    } while (0);

#define ROW_ID_MASK 0xFFFFFFFFLL