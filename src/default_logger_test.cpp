
#include "./default_logger.h"

#include <catch2/catch_test_macros.hpp>

#include "vsag/logger.h"

TEST_CASE("test default logger", "[ut][logger]") {
    vsag::DefaultLogger logger;
    logger.SetLevel(vsag::Logger::Level::kTRACE);
    logger.Trace("this is a trace level message");
    logger.Debug("this is a debug level message");
    logger.Info("this is a info level message");
    logger.Warn("this is a warn level message");
    logger.Error("this is a error level message");
    logger.Critical("this is a critical level message");
}
