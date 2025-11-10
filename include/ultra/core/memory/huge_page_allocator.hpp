#pragma once
#include "../compiler.hpp"
#include <cstddef>
#include <sys/mman.h>
#include <iostream>

namespace ultra {

/**
 * Allocator using huge pages (2MB) for reduced TLB misses
 * Critical for order book and market data buffers
 * (Note: Implementation is in the header as it's a template)
 */
template<typename T>
class HugePageAllocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    constexpr static size_type HUGE_PAGE_SIZE = 2 * 1024 * 1024; // 2MB
    
    HugePageAllocator() noexcept = default;
    
    template<typename U>
    HugePageAllocator(const HugePageAllocator<U>&) noexcept {}
    
    T* allocate(size_type n) {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        
        size_type num_bytes = n * sizeof(T);
        // Align allocation to huge page size
        size_type aligned_bytes = ((num_bytes + HUGE_PAGE_SIZE - 1) / HUGE_PAGE_SIZE) * HUGE_PAGE_SIZE;
        
        void* p = mmap(nullptr, 
                       aligned_bytes, 
                       PROT_READ | PROT_WRITE, 
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, 
                       -1, 0);
                       
        if (p == MAP_FAILED) {
            // Fallback to standard mmap without huge pages
            p = mmap(nullptr, 
                     aligned_bytes, 
                     PROT_READ | PROT_WRITE, 
                     MAP_PRIVATE | MAP_ANONYMOUS, 
                     -1, 0);
            if (p == MAP_FAILED) {
                throw std::bad_alloc();
            }
        }
        return static_cast<T*>(p);
    }
    
    void deallocate(T* p, size_type n) noexcept {
        if (p == nullptr) return;
        
        size_type num_bytes = n * sizeof(T);
        size_type aligned_bytes = ((num_bytes + HUGE_PAGE_SIZE - 1) / HUGE_PAGE_SIZE) * HUGE_PAGE_SIZE;
        munmap(p, aligned_bytes);
    }
    
    template<typename U>
    struct rebind {
        using other = HugePageAllocator<U>;
    };
};

} // namespace ultra
