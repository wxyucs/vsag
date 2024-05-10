#pragma once

#include <memory>
#include <string>

namespace vsag {

class Logger;
using LoggerPtr = std::shared_ptr<Logger>;

class Logger {
public:
    enum Level : int {
        TRACE = 0,
        DEBUG = 1,
        INFO = 2,
        WARN = 3,
        ERR = 4,
        CRITICAL = 5,
        OFF = 6,
        N_LEVELS
    };

    virtual void
    SetLevel(Level log_level) = 0;

    virtual void
    Trace(const std::string& msg) = 0;

    virtual void
    Debug(const std::string& msg) = 0;

    virtual void
    Info(const std::string& msg) = 0;

    virtual void
    Warn(const std::string& msg) = 0;

    virtual void
    Error(const std::string& msg) = 0;

    virtual void
    Critical(const std::string& msg) = 0;

public:
    virtual ~Logger() = default;
};

}  // namespace vsag
