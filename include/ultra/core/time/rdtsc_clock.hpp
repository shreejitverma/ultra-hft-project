#pragma once
#include "../compiler.hpp"
#include "../types.hpp"
#include <x86intrin.h>
#include <chrono>
#include <atomic>

namespace ultra {

/**
 * Ultra-low latency clock using RDTSC instruction
 * Calibrated to nanoseconds, ~20ns overhead
 */
class RDTSCClock {
public:
    static void calibrate() noexcept;
    
    // Get current time in nanoseconds (fastest path)
    ULTRA_ALWAYS_INLINE static Timestamp now() noexcept {
        return rdtsc_to_ns(__rdtsc());
    }
    
    // Get raw TSC value
    ULTRA_ALWAYS_INLINE static uint64_t rdtsc() noexcept {
        return __rdtsc();
    }
    
    // Serializing RDTSC (more expensive but precise)
    ULTRA_ALWAYS_INLINE static uint64_t rdtscp() noexcept {
        uint32_t aux;
        return __rdtscp(&aux);
    }
    
    // Convert TSC to nanoseconds
    ULTRA_ALWAYS_INLINE static Timestamp rdtsc_to_ns(uint64_t tsc) noexcept {
        return static_cast<Timestamp>(tsc * tsc_to_ns_factor_.load(std::memory_order_relaxed));
    }
    
    // Get system time (slower, for logging)
    static Timestamp system_now() noexcept;
    
private:
    static std::atomic<double> tsc_to_ns_factor_;
};

} // namespace ultra
