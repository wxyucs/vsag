//
// Created by jinjiabao.jjb on 2/16/24.
//

#include "vsag/options.h"

#include <utility>

#include "default_allocator.h"
#include "default_logger.h"
#include "logger.h"

namespace vsag {

Options&
Options::Instance() {
    static Options s_instance;
    return s_instance;
}

Allocator*
Options::allocator() {
    if (not global_allocator_) {
        this->set_allocator(std::make_unique<DefaultAllocator>());
    }
    return global_allocator_.get();
}

LoggerPtr
Options::logger() {
    if (not logger_) {
        this->set_logger(std::shared_ptr<Logger>(new DefaultLogger()));
    }
    return logger_;
}

}  // namespace vsag
