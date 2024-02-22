#pragma once

#include <atomic>
#include <string>

namespace vsag {

class Option {
private:
    // Private constructor and destructor to ensure singleton
    Option();
    ~Option();

    // Deleted copy constructor and assignment operator to prevent copies
    Option(const Option&) = delete;
    Option&
    operator=(const Option&) = delete;

    // In a single query, the space size used to store disk vectors.
    std::atomic<size_t> sector_size_;

public:
    static Option&
    Instance();

    size_t
    GetSectorSize() const;

    void
    SetSectorSize(size_t size);
};

}  // namespace vsag
