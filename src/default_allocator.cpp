//
// Created by root on 4/19/24.
//

#include "default_allocator.h"

#include "vsag/option.h"

namespace vsag {

void*
allocate(size_t size) {
    return Option::Instance().GetAllocator()->Allocate(size);
}

void
deallocate(void* p) {
    Option::Instance().GetAllocator()->Deallocate(p);
}

void*
reallocate(void* p, size_t size) {
    return Option::Instance().GetAllocator()->Reallocate(p, size);
}

void*
DefaultAllocator::Allocate(size_t size) {
    return malloc(size);
}
void
DefaultAllocator::Deallocate(void* p) {
    free(p);
}

void*
DefaultAllocator::Reallocate(void* p, size_t size) {
    return realloc(p, size);
}

std::string
DefaultAllocator::Name() {
    return "DefaultAllocator";
}

}  // namespace vsag
