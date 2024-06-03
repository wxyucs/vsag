//
// Created by jinjiabao.jjb on 2/16/24.
//

#include "vsag/options.h"

#include <utility>

#include "default_allocator.h"
#include "default_logger.h"
#include "logger.h"

namespace vsag {

static std::shared_ptr<DefaultAllocator> _default_allocator = std::make_shared<DefaultAllocator>();

Options&
Options::Instance() {
    static Options s_instance;
    return s_instance;
}

Allocator*
Options::allocator() {
    if (not global_allocator_) {
        this->set_allocator(_default_allocator.get());
    }
    return global_allocator_;
}

LoggerPtr
Options::logger() {
    if (not logger_) {
        this->set_logger(std::shared_ptr<Logger>(new DefaultLogger()));
    }
    return logger_;
}

}  // namespace vsag
