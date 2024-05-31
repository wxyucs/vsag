
#include <iostream>

#include "vsag/vsag.h"

class ExampleAllocator : public vsag::Allocator {
public:
    std::string
    Name() override {
        return "myallocator";
    }

    void*
    Allocate(size_t size) override {
        std::cout << Name() << ": allocate " << size << " bytes" << std::endl;
        return malloc(size);
    }

    void
    Deallocate(void* p) override {
        std::cout << Name() << ": deallocate " << p << std::endl;
        return free(p);
    }

    void*
    Reallocate(void* p, size_t size) override {
        std::cout << Name() << ": reallocate " << p << " with " << size << " size" << std::endl;
        return realloc(p, size);
    }
};

int
main() {
    // set allocator before creating any index
    vsag::Options::Instance().set_allocator(std::make_unique<ExampleAllocator>());

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
    std::cout << "create index" << std::endl;
    auto index = vsag::Factory::CreateIndex("hnsw", paramesters);

    std::cout << "delete index" << std::endl;
    index = nullptr;

    return 0;
}
