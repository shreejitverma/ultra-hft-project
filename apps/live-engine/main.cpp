#include "engine.hpp"
#include "ultra/core/time/rdtsc_clock.hpp"
#include <iostream>

int main() {
    std::cout << "Starting Ultra-Low Latency Trading Engine..." << std::endl;

    // --- 1. Calibrate Clock ---
    // This is critical. Do it once at startup.
    ultra::RDTSCClock::calibrate();

    // --- 2. Pin Threads ---
    // (A real app would use sched_setaffinity here to pin
    // md_thread, strategy_thread, etc. to isolated cores
    // from config/engine.toml)
    
    // --- 3. Allocate Memory ---
    // (A real app would pre-allocate all memory,
    // pool allocators, and huge pages here)

    // --- 4. Create and Run Engine ---
    try {
        ultra::Engine engine;
        engine.run();
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Engine shutdown complete." << std::endl;
    return 0;
}
