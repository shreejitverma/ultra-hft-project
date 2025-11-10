#include "ultra/core/time/rdtsc_clock.hpp"
#include <thread>
#include <iostream>

namespace ultra {

// Initialize statics
std::atomic<double> RDTSCClock::tsc_to_ns_factor_{0.0};

void RDTSCClock::calibrate() noexcept {
    std::cout << "Calibrating RDTSCClock..." << std::endl;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    uint64_t r1 = rdtsc();
    
    // Sleep for a non-trivial amount of time
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto t2 = std::chrono::high_resolution_clock::now();
    uint64_t r2 = rdtsc();
    
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    uint64_t rdtsc_ticks = r2 - r1;
    
    double factor = static_cast<double>(duration_ns) / static_cast<double>(rdtsc_ticks);
    tsc_to_ns_factor_.store(factor, std::memory_order_release);
    
    std::cout << "RDTSC Ticks: " << rdtsc_ticks << std::endl;
    std::cout << "Nanoseconds: " << duration_ns << " ns" << std::endl;
    std::cout << "Factor (ns/tick): " << factor << std::endl;
    std::cout << "Calibration complete." << std::endl;
}

Timestamp RDTSCClock::system_now() noexcept {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

} // namespace ultra
