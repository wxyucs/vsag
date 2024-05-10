
#include "./default_logger.h"

namespace vsag {

void
DefaultLogger::SetLevel(Logger::Level log_level) {
    spdlog::set_level((spdlog::level::level_enum)log_level);
}

void
DefaultLogger::Trace(const std::string& msg) {
    spdlog::trace(msg);
}

void
DefaultLogger::Debug(const std::string& msg) {
    spdlog::debug(msg);
}

void
DefaultLogger::Info(const std::string& msg) {
    spdlog::info(msg);
}

void
DefaultLogger::Warn(const std::string& msg) {
    spdlog::warn(msg);
}

void
DefaultLogger::Error(const std::string& msg) {
    spdlog::error(msg);
}

void
DefaultLogger::Critical(const std::string& msg) {
    spdlog::critical(msg);
}

}  // namespace vsag
