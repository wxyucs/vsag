
#include <iostream>

#include "vsag/vsag.h"

// custom logger
class MyLogger : public vsag::Logger {
public:
    inline void
    SetLevel(Level log_level) override {
        level_ = log_level - vsag::Logger::Level::kTRACE;
    }

    inline void
    Trace(const std::string& msg) override {
        if (level_ <= 0) {
            std::cout << "[mylogger]::[trace] " << msg << std::endl;
        }
    }

    inline void
    Debug(const std::string& msg) override {
        if (level_ <= 1) {
            std::cout << "[mylogger]::[debug] " << msg << std::endl;
        }
    }

    inline void
    Info(const std::string& msg) override {
        if (level_ <= 2) {
            std::cout << "[mylogger]::[info] " << msg << std::endl;
        }
    }

    inline void
    Warn(const std::string& msg) override {
        if (level_ <= 3) {
            std::cout << "[mylogger]::[warn] " << msg << std::endl;
        }
    }

    inline void
    Error(const std::string& msg) override {
        if (level_ <= 4) {
            std::cout << "[mylogger]::[error] " << msg << std::endl;
        }
    }

    void
    Critical(const std::string& msg) override {
        if (level_ <= 5) {
            std::cout << "[mylogger]::[critical] " << msg << std::endl;
        }
    }

    int64_t level_ = 0;
};

int
main() {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kDEBUG);

    // do something
    {
        auto paramesters = R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 4,
            "hnsw": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        )";
        auto index = vsag::Factory::CreateIndex("hnsw", paramesters);
    }

    vsag::Options::Instance().set_logger(vsag::LoggerPtr(new MyLogger()));
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kDEBUG);

    // do something
    {
        auto paramesters = R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 4,
            "hnsw": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        )";
        auto index = vsag::Factory::CreateIndex("hnsw", paramesters);
    }
}
