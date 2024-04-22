#pragma once

#include <mutex>
#include <string.h>
#include <deque>

namespace vsag {

extern void*
allocate(size_t size);

extern void
deallocate(void* p);

extern void*
reallocate(void* p, size_t size);

} // namespace vsag


namespace hnswlib {
typedef unsigned short int vl_type;

class VisitedList {
 public:
    vl_type curV;
    vl_type *mass;
    unsigned int numelements;

    VisitedList(int numelements1) {
        curV = -1;
        numelements = numelements1;
        mass = (vl_type *) vsag::allocate(numelements * sizeof(vl_type));
    }

    void reset() {
        curV++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;
        }
    }

    ~VisitedList() { vsag::deallocate(mass); }
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
    std::deque<VisitedList *> pool;
    std::mutex poolguard;
    int numelements;

 public:
    VisitedListPool(int initmaxpools, int numelements1) {
        numelements = numelements1;
        for (int i = 0; i < initmaxpools; i++)
            pool.push_front(new VisitedList(numelements));
    }

    VisitedList *getFreeVisitedList() {
        VisitedList *rez;
        {
            std::unique_lock <std::mutex> lock(poolguard);
            if (pool.size() > 0) {
                rez = pool.front();
                pool.pop_front();
            } else {
                rez = new VisitedList(numelements);
            }
        }
        rez->reset();
        return rez;
    }

    void releaseVisitedList(VisitedList *vl) {
        std::unique_lock <std::mutex> lock(poolguard);
        pool.push_front(vl);
    }

    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList *rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }
};
}  // namespace hnswlib
