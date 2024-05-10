
#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "vsag/options.h"

void
info(const std::string& msg) {
    vsag::Options::Instance().logger()->Info(msg);
}

TEST_CASE("set external logger", "[ft][options]") {
    class MyLogger : public vsag::Logger {
    public:
        inline void
        SetLevel(Level log_level) override {
            level_ = log_level - vsag::Logger::Level::TRACE;
        }

        inline void
        Trace(const std::string& msg) override {
            if (level_ <= 0) {
                std::cout << "[trace]" << msg << std::endl;
            }
        }

        inline void
        Debug(const std::string& msg) override {
            if (level_ <= 1) {
                std::cout << "[debug]" << msg << std::endl;
            }
        }

        inline void
        Info(const std::string& msg) override {
            if (level_ <= 2) {
                std::cout << "[info]" << msg << std::endl;
            }
        }

        inline void
        Warn(const std::string& msg) override {
            if (level_ <= 3) {
                std::cout << "[warn]" << msg << std::endl;
            }
        }

        inline void
        Error(const std::string& msg) override {
            if (level_ <= 4) {
                std::cout << "[error]" << msg << std::endl;
            }
        }

        void
        Critical(const std::string& msg) override {
            if (level_ <= 5) {
                std::cout << "[critical]" << msg << std::endl;
            }
        }

        int64_t level_ = 0;
    };

    info("test test, by default logger");
    vsag::Options::Instance().set_logger(vsag::LoggerPtr(new MyLogger()));
    info("test test, by my logger");
}
