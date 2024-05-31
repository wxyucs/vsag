#pragma once

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include "vsag/logger.h"
#include "vsag/options.h"

namespace vsag {
namespace logger {

enum class level {
    trace = Logger::Level::kTRACE,
    debug = Logger::Level::kDEBUG,
    info = Logger::Level::kINFO,
    warn = Logger::Level::kWARN,
    err = Logger::Level::kERR,
    critical = Logger::Level::kCRITICAL,
    off = Logger::Level::kOFF
};

inline void
set_level(level log_level) {
    Options::Instance().logger()->SetLevel((Logger::Level)log_level);
}

inline void
trace(const std::string& msg) {
    Options::Instance().logger()->Trace(msg);
}

inline void
debug(const std::string& msg) {
    Options::Instance().logger()->Debug(msg);
}

inline void
info(const std::string& msg) {
    Options::Instance().logger()->Info(msg);
}

inline void
warn(const std::string& msg) {
    Options::Instance().logger()->Warn(msg);
}

inline void
error(const std::string& msg) {
    Options::Instance().logger()->Error(msg);
}

inline void
critical(const std::string& msg) {
    Options::Instance().logger()->Critical(msg);
}

template <typename... Args>
inline void
trace(fmt::format_string<Args...> fmt, Args&&... args) {
    trace(fmt::format(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
inline void
debug(fmt::format_string<Args...> fmt, Args&&... args) {
    debug(fmt::format(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
inline void
info(fmt::format_string<Args...> fmt, Args&&... args) {
    info(fmt::format(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
inline void
warn(fmt::format_string<Args...> fmt, Args&&... args) {
    warn(fmt::format(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
inline void
error(fmt::format_string<Args...> fmt, Args&&... args) {
    error(fmt::format(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
inline void
critical(fmt::format_string<Args...> fmt, Args&&... args) {
    critical(fmt::format(fmt, std::forward<Args>(args)...));
}

}  // namespace logger
}  // namespace vsag
