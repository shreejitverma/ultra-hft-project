#pragma once
#include "../compiler.hpp"
#include <atomic>
#include <array>
#include <type_traits>

namespace ultra {

/**
 * Single-Producer Single-Consumer lock-free ring buffer
 * Optimized for market data pipelines
 * False-sharing prevention with cache line padding
 */
template<typename T, size_t Capacity>
requires std::is_trivially_copyable_v<T>
class SPSCQueue {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    
    SPSCQueue() noexcept : head_(0), tail_(0) {}
    
    // Producer: push (returns false if full)
    ULTRA_ALWAYS_INLINE bool push(const T& item) noexcept {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next = (head + 1) & MASK;
        
        if (ULTRA_UNLIKELY(next == tail_.load(std::memory_order_acquire))) {
            return false; // Queue full
        }
        
        buffer_[head] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }
    
    // Consumer: pop (returns false if empty)
    ULTRA_ALWAYS_INLINE bool pop(T& item) noexcept {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        
        if (ULTRA_UNLIKELY(tail == head_.load(std::memory_order_acquire))) {
            return false; // Queue empty
        }
        
        item = buffer_[tail];
        tail_.store((tail + 1) & MASK, std::memory_order_release);
        return true;
    }
    
    ULTRA_ALWAYS_INLINE bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }
    
    ULTRA_ALWAYS_INLINE size_t size() const noexcept {
        return (head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire)) & MASK;
    }
    
private:
    static constexpr size_t MASK = Capacity - 1;
    
    std::array<T, Capacity> buffer_;
    
    // Cache line padding to prevent false sharing
    ULTRA_CACHE_ALIGNED std::atomic<size_t> head_;
    ULTRA_CACHE_ALIGNED std::atomic<size_t> tail_;
};

} // namespace ultra
