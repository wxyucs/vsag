#pragma once

#include <spdlog/spdlog.h>

#include "vsag/logger.h"
#include "vsag/options.h"

namespace vsag {

class DefaultLogger : public Logger {
public:
    void
    SetLevel(Logger::Level log_level) override;

    void
    Trace(const std::string& msg) override;

    void
    Debug(const std::string& msg) override;

    void
    Info(const std::string& msg) override;

    void
    Warn(const std::string& msg) override;

    void
    Error(const std::string& msg) override;

    void
    Critical(const std::string& msg) override;
};

}  // namespace vsag
