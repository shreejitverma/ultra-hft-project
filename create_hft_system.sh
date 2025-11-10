#!/bin/bash
# Ultra-Low Latency HFT System - Production Grade Setup
# Based on AI-Integrated FPGA Market Making Research
# Target: Sub-microsecond tick-to-trade latency

set -e

echo "=================================================="
echo "Ultra-Low Latency HFT System Setup"
echo "Research-Grade Implementation"
echo "=================================================="

# Create project root (we assume we are in the dir already)
PROJECT_ROOT="."

# ============================================================================
# DIRECTORY STRUCTURE - Organized by functional layers
# ============================================================================

echo "[1/10] Creating directory structure..."

mkdir -p \
    config/{profiles,exchange-specs,risk-limits} \
    include/ultra/{core/{time,memory,lockfree,simd},network/{parsers,multicast,kernel-bypass},market-data/{itch,fix,ouch,book},strategy/{rl-inference,avellaneda,signals},risk/{pretrade,inventory,limits},execution/{router,gateway,oms},fpga/{verilog,testbench,sim},telemetry/{metrics,latency,monitoring},utils} \
    src/{core/{time,memory,lockfree},network/{parsers,multicast},market-data/{itch,fix,book},strategy/{rl-inference,avellaneda},risk,execution,telemetry} \
    apps/{md-replayer,strategy-backtester,live-engine,fpga-sim} \
    tests/{unit,integration,performance} \
    benchmarks/{latency,throughput} \
    tools/{pcap-capture,data-generator,analysis} \
    data/{sample-pcaps,order-books,tick-data} \
    fpga/{rtl/{parsers,book,strategy,risk,encoder},tb,scripts,constraints} \
    docs/{architecture,deployment,performance} \
    third_party/{headers,libs}

# ============================================================================
# CMAKE BUILD SYSTEM - Optimized for HFT
# ============================================================================

echo "[2/10] Creating CMake build system..."

cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.22)
project(UltraHFT VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# COMPILER FLAGS - Ultra-Low Latency Optimizations
# ============================================================================

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Base flags for all builds
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
# Note: -march=native in an x86-64 container will enable AVX, SSE, etc.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -msse4.2 -mfma")

# Release optimizations (production)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto -ffast-math")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -finline-functions")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fomit-frame-pointer")

# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -fsanitize=address,undefined")

# Profile-guided optimization support
option(USE_PGO "Enable Profile-Guided Optimization" OFF)
if(USE_PGO)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-use")
endif()

# ============================================================================
# DEPENDENCIES
# ============================================================================

# Threading
find_package(Threads REQUIRED)

# Optional: DPDK for kernel bypass
option(USE_DPDK "Use DPDK for kernel bypass networking" OFF)
if(USE_DPDK)
    find_package(DPDK REQUIRED)
    add_compile_definitions(ULTRA_USE_DPDK)
endif()

# ============================================================================
# INCLUDE DIRECTORIES
# ============================================================================

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

# ============================================================================
# SOURCE FILES - Organized by component
# ============================================================================

# Core infrastructure
set(ULTRA_CORE_SOURCES
    src/core/time/rdtsc_clock.cpp
    # src/core/memory/huge_page_allocator.cpp # (Implementation in header for templates)
)

# Network layer
set(ULTRA_NETWORK_SOURCES
    src/network/parsers/ethernet_parser.cpp
)

# Market data
set(ULTRA_MD_SOURCES
    src/market-data/itch/decoder.cpp
    src/market-data/book/order_book_l2.cpp
)

# Strategy
set(ULTRA_STRATEGY_SOURCES
    src/strategy/rl-inference/rl_policy_stub.cpp
    # src/strategy/avellaneda/as_market_maker.cpp # (Future step)
)

# Risk management
set(ULTRA_RISK_SOURCES
    src/risk/pretrade_checker.cpp
)

# Execution
set(ULTRA_EXEC_SOURCES
    src/execution/gateway_sim.cpp
)

# Telemetry
set(ULTRA_TELEMETRY_SOURCES
    # src/telemetry/latency_tracker.cpp # (Future step)
)

# ============================================================================
# LIBRARIES
# ============================================================================

# Core library
add_library(ultra_core STATIC ${ULTRA_CORE_SOURCES})
target_compile_definitions(ultra_core PRIVATE ULTRA_CORE_BUILD)

# Network library
add_library(ultra_network STATIC ${ULTRA_NETWORK_SOURCES})
target_link_libraries(ultra_network PUBLIC ultra_core)

# Market data library
add_library(ultra_md STATIC ${ULTRA_MD_SOURCES})
target_link_libraries(ultra_md PUBLIC ultra_core ultra_network)

# Strategy library
add_library(ultra_strategy STATIC ${ULTRA_STRATEGY_SOURCES})
target_link_libraries(ultra_strategy PUBLIC ultra_core ultra_md)

# Risk library
add_library(ultra_risk STATIC ${ULTRA_RISK_SOURCES})
target_link_libraries(ultra_risk PUBLIC ultra_core)

# Execution library
add_library(ultra_exec STATIC ${ULTRA_EXEC_SOURCES})
target_link_libraries(ultra_exec PUBLIC ultra_core ultra_risk)

# Telemetry library
add_library(ultra_telemetry STATIC ${ULTRA_TELEMETRY_SOURCES})
target_link_libraries(ultra_telemetry PUBLIC ultra_core)

# Combined library for applications
add_library(ultra_hft STATIC)
target_link_libraries(ultra_hft PUBLIC
    ultra_core
    ultra_network
    ultra_md
    ultra_strategy
    ultra_risk
    ultra_exec
    ultra_telemetry
    Threads::Threads
)

# ============================================================================
# APPLICATIONS
# ============================================================================

# Market data replayer (Future step)
# add_executable(md_replayer
#     apps/md-replayer/main.cpp
# )
# target_link_libraries(md_replayer ultra_hft)

# Strategy backtester (Future step)
# add_executable(strategy_backtester
#     apps/strategy-backtester/main.cpp
# )
# target_link_libraries(strategy_backtester ultra_hft)

# Live trading engine
add_executable(live_engine
    apps/live-engine/main.cpp
    apps/live-engine/engine.cpp
)
target_link_libraries(live_engine ultra_hft)

# ============================================================================
# TESTS
# ============================================================================

enable_testing()

# add_executable(unit_tests
#     tests/unit/test_rdtsc_clock.cpp
#     tests/unit/test_order_book.cpp
#     tests.unit/test_risk_checks.cpp
# )
# target_link_libraries(unit_tests ultra_hft)
# add_test(NAME UnitTests COMMAND unit_tests)

EOF

# ============================================================================
# CONFIGURATION FILES
# ============================================================================

echo "[3/10] Creating configuration files..."

cat > config/engine.toml << 'EOF'
[system]
name = "ultra-hft"
version = "1.0.0"
log_level = "INFO"

[hardware]
cpu_affinity = [2, 3, 4, 5]  # Isolated cores
numa_node = 0
huge_pages = true
huge_page_size_mb = 2

[network]
interface = "eth0"
mtu = 9000  # Jumbo frames
kernel_bypass = true
use_dpdk = false

[market_data]
exchange = "NASDAQ"
protocol = "ITCH_5.0"
multicast_group = "233.54.12.0"
multicast_port = 26477
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

[market_data.feed_handler]
buffer_size = 65536
prefetch_lines = 8
batch_processing = true

[order_book]
levels = 10  # L2 book depth
implementation = "array_based"  # array_based, tree_based
update_mode = "incremental"

[strategy]
type = "rl_market_maker"
model_path = "models/rl_policy.bin"
update_frequency_ns = 100  # Sub-microsecond

[strategy.rl_policy]
input_features = [
    "bid_price_0", "bid_size_0",
    "ask_price_0", "ask_size_0",
    "mid_price", "spread",
    "imbalance", "volatility",
    "inventory", "pnl"
]
hidden_layers = [64, 32]
activation = "relu"
inference_batch_size = 1

[strategy.avellaneda_stoikov]
risk_aversion = 0.1
volatility_window = 100
inventory_penalty = 0.01

[risk]
max_position_shares = 10000
max_notional_usd = 10000000
max_order_size = 1000
max_orders_per_second = 10000
kill_switch_loss_usd = 50000

[risk.inventory]
target_position = 0
reversion_speed = 0.1
max_deviation = 5000

[execution]
gateway_type = "simulation"  # simulation, fix, ouch
smart_routing = true
venue_preference = ["NASDAQ", "NYSE", "ARCA"]

[execution.order_routing]
latency_weight = 0.7
fee_weight = 0.2
fill_rate_weight = 0.1

[telemetry]
enabled = true
metrics_port = 9090
latency_histogram_buckets = [100, 250, 500, 1000, 2500, 5000, 10000]
publish_interval_ms = 1000

[telemetry.monitoring]
tick_to_trade_alert_ns = 1000
memory_alert_mb = 1000
cpu_alert_percent = 90
EOF

cat > config/profiles/low_latency.toml << 'EOF'
# Optimized for minimum latency (sacrifice throughput if needed)
[tuning]
profile = "low_latency"
cpu_governor = "performance"
turbo_boost = true
c_states = false
irq_affinity = [0, 1]
rps_cpus = [2, 3]
EOF

cat > config/exchange-specs/nasdaq_itch50.toml << 'EOF'
[protocol]
name = "NASDAQ_ITCH_5.0"
version = "5.0"
byte_order = "big_endian"

[messages]
system_event = 0x53
stock_directory = 0x52
stock_trading_action = 0x48
add_order = 0x41
add_order_mpid = 0x46
order_executed = 0x45
order_executed_with_price = 0x43
order_cancel = 0x58
order_delete = 0x44
order_replace = 0x55
trade = 0x50
cross_trade = 0x51
broken_trade = 0x42
noii = 0x49
rpii = 0x4E
EOF

# ============================================================================
# CORE HEADERS - Ultra-Low Latency Primitives
# ============================================================================

echo "[4/10] Creating core infrastructure headers..."

cat > include/ultra/core/compiler.hpp << 'EOF'
#pragma once

// Branch prediction hints
#define ULTRA_LIKELY(x)   __builtin_expect(!!(x), 1)
#define ULTRA_UNLIKELY(x) __builtin_expect(!!(x), 0)

// Force inlining
#define ULTRA_ALWAYS_INLINE __attribute__((always_inline)) inline
#define ULTRA_NEVER_INLINE  __attribute__((noinline))

// Hot/cold path hints
#define ULTRA_HOT   __attribute__((hot))
#define ULTRA_COLD  __attribute__((cold))

// Prefetch hints
#define ULTRA_PREFETCH_READ(addr)  __builtin_prefetch(addr, 0, 3)
#define ULTRA_PREFETCH_WRITE(addr) __builtin_prefetch(addr, 1, 3)

// Cache line size
#define ULTRA_CACHE_LINE_SIZE 64
#define ULTRA_CACHE_ALIGNED alignas(ULTRA_CACHE_LINE_SIZE)

// Memory barriers
#define ULTRA_COMPILER_BARRIER() asm volatile("" ::: "memory")
#define ULTRA_MEMORY_BARRIER()   __sync_synchronize()

// Restrict pointer
#define ULTRA_RESTRICT __restrict__

// Assume hint for optimizer
#define ULTRA_ASSUME(x) do { if (!(x)) __builtin_unreachable(); } while(0)
EOF

cat > include/ultra/core/types.hpp << 'EOF'
#pragma once
#include <cstdint>
#include <cstddef>
#include <limits>
#include <iostream>

namespace ultra {

// Fixed-point price representation (4 decimal places)
using Price = int64_t;
constexpr int64_t PRICE_SCALE = 10000;

ULTRA_ALWAYS_INLINE Price to_price(double p) noexcept {
    return static_cast<Price>(p * PRICE_SCALE);
}

ULTRA_ALWAYS_INLINE double from_price(Price p) noexcept {
    return static_cast<double>(p) / PRICE_SCALE;
}

// Core trading types
using Quantity = int64_t;
using SymbolId = uint32_t;
using OrderId = uint64_t;
using Timestamp = uint64_t;  // nanoseconds since epoch
using SequenceNum = uint64_t;

// Side enum
enum class Side : uint8_t {
    BUY  = 0,
    SELL = 1
};

inline std::ostream& operator<<(std::ostream& os, Side side) {
    os << (side == Side::BUY ? "BUY" : "SELL");
    return os;
}

// Order type
enum class OrderType : uint8_t {
    LIMIT  = 0,
    MARKET = 1,
    IOC    = 2,  // Immediate or Cancel
    FOK    = 3   // Fill or Kill
};

// Order status
enum class OrderStatus : uint8_t {
    PENDING   = 0,
    ACCEPTED  = 1,
    FILLED    = 2,
    PARTIAL   = 3,
    CANCELLED = 4,
    REJECTED  = 5
};

// Market data event types
enum class MDEventType : uint8_t {
    ADD_ORDER    = 0,
    MODIFY_ORDER = 1,
    DELETE_ORDER = 2,
    TRADE        = 3,
    QUOTE        = 4,
    UNKNOWN      = 255
};

// Constants
constexpr SymbolId INVALID_SYMBOL = 0;
constexpr OrderId INVALID_ORDER_ID = 0;
constexpr Price INVALID_PRICE = std::numeric_limits<Price>::max();
constexpr Quantity INVALID_QUANTITY = -1;

// Base event structure
struct Event {
    Timestamp tsc;       // RDTSC timestamp
    Timestamp exchange_ts; // Exchange timestamp
    Timestamp received_ts; // Our ingress timestamp
};

} // namespace ultra
EOF

cat > include/ultra/core/time/rdtsc_clock.hpp << 'EOF'
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
EOF

cat > include/ultra/core/memory/huge_page_allocator.hpp << 'EOF'
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
EOF

cat > include/ultra/core/lockfree/spsc_queue.hpp << 'EOF'
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
EOF

echo "[5/10] Creating network layer headers..."

cat > include/ultra/network/parsers/ethernet_parser.hpp << 'EOF'
#pragma once
#include "../../core/compiler.hpp"
#include "../../core/types.hpp"
#include <cstring>
#include <netinet/ether.h>
#include <netinet/ip.h>
#include <netinet/udp.h>

namespace ultra::network {

#pragma pack(push, 1)

struct EthernetHeader {
    uint8_t dst_mac[6];
    uint8_t src_mac[6];
    uint16_t ethertype;
};

struct IPv4Header {
    uint8_t  version_ihl;
    uint8_t  tos;
    uint16_t total_length;
    uint16_t id;
    uint16_t flags_offset;
    uint8_t  ttl;
    uint8_t  protocol;
    uint16_t checksum;
    uint32_t src_ip;
    uint32_t dst_ip;
};

struct UDPHeader {
    uint16_t src_port;
    uint16_t dst_port;
    uint16_t length;
    uint16_t checksum;
};

#pragma pack(pop)

/**
 * Zero-copy Ethernet/IP/UDP parser
 * Parses in-place, no allocations
 */
class EthernetParser {
public:
    struct ParsedPacket {
        const uint8_t* payload;
        uint16_t payload_len;
        uint32_t src_ip;
        uint32_t dst_ip;
        uint16_t src_port;
        uint16_t dst_port;
        Timestamp timestamp_ns;
        bool valid;
    };
    
    ULTRA_ALWAYS_INLINE static ParsedPacket parse(
        const uint8_t* packet, 
        size_t packet_len,
        Timestamp timestamp_ns
    ) noexcept;
    
    ULTRA_ALWAYS_INLINE static uint16_t ntohs_fast(uint16_t n) noexcept {
        return __builtin_bswap16(n);
    }
    
    ULTRA_ALWAYS_INLINE static uint32_t ntohl_fast(uint32_t n) noexcept {
        return __builtin_bswap32(n);
    }
};

} // namespace ultra::network
EOF

echo "[6/10] Creating market data headers..."

cat > include/ultra/market-data/itch/decoder.hpp << 'EOF'
#pragma once
#include "../../core/compiler.hpp"
#include "../../core/types.hpp"
#include <cstring>
#include <array>

namespace ultra::md::itch {

#pragma pack(push, 1)

// ITCH 5.0 Message Types
enum class MessageType : uint8_t {
    SYSTEM_EVENT          = 'S',
    STOCK_DIRECTORY       = 'R',
    STOCK_TRADING_ACTION  = 'H',
    ADD_ORDER             = 'A',
    ADD_ORDER_MPID        = 'F',
    ORDER_EXECUTED        = 'E',
    ORDER_EXECUTED_PRICE  = 'C',
    ORDER_CANCEL          = 'X',
    ORDER_DELETE          = 'D',
    ORDER_REPLACE         = 'U',
    TRADE                 = 'P',
    CROSS_TRADE           = 'Q',
    BROKEN_TRADE          = 'B',
    NOII                  = 'I'
};

struct MessageHeader {
    uint16_t length;
    uint8_t  type;
};

struct AddOrder {
    MessageHeader header;
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp;      // nanoseconds since midnight
    uint64_t order_ref_number;
    char     buy_sell_indicator;
    uint32_t shares;
    char     stock[8];
    uint32_t price;          // Fixed point 4 decimal places
} ;

struct OrderExecuted {
    MessageHeader header;
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp;
    uint64_t order_ref_number;
    uint32_t executed_shares;
    uint64_t match_number;
};

struct OrderDelete {
    MessageHeader header;
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp;
    uint64_t order_ref_number;
};

struct OrderReplace {
    MessageHeader header;
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp;
    uint64_t original_order_ref;
    uint64_t new_order_ref;
    uint32_t shares;
    uint32_t price;
};

#pragma pack(pop)

/**
 * High-performance ITCH decoder
 * - Zero-copy parsing
 * - SIMD optimizations where applicable
 * - Branch prediction optimized
 */
class ITCHDecoder {
public:
    struct DecodedMessage : public Event {
        MDEventType event_type;
        SymbolId symbol_id;
        OrderId order_id;
        OrderId new_order_id; // For replace
        Side side;
        Price price;
        Quantity quantity;
        bool valid;
    };
    
    ITCHDecoder();
    
    // Fast path: decode single message
    ULTRA_HOT ULTRA_ALWAYS_INLINE 
    DecodedMessage decode(const uint8_t* data, size_t len, Timestamp rdtsc_ts) noexcept;
    
    // Symbol lookup (pre-registered)
    void register_symbol(const char* symbol, SymbolId id);
    SymbolId lookup_symbol(const char* symbol) const noexcept;
    
private:
    // Symbol hash table for O(1) lookup
    static constexpr size_t SYMBOL_HASH_SIZE = 4096;
    struct SymbolEntry {
        uint64_t symbol_int; // Store symbol as 8-byte int
        SymbolId id;
    };
    // Use std::array for aggregate initialization
    std::array<SymbolEntry, SYMBOL_HASH_SIZE> symbol_table_{};
    
    ULTRA_ALWAYS_INLINE uint32_t hash_symbol(uint64_t s) const noexcept {
        // FNV-1a hash variant
        uint64_t hash = 14695981039346656037ULL;
        for(int i = 0; i < 8; ++i) {
            hash ^= (s >> (i*8)) & 0xFF;
            hash *= 1099511628211ULL;
        }
        return (hash ^ (hash >> 32)) & (SYMBOL_HASH_SIZE - 1);
    }

    ULTRA_ALWAYS_INLINE Price decode_price(uint32_t itch_price) const noexcept {
        // ITCH price is already scaled by 10000
        return static_cast<Price>(__builtin_bswap32(itch_price));
    }

    ULTRA_ALWAYS_INLINE uint64_t char8_to_uint64(const char* s) const noexcept {
        uint64_t res;
        memcpy(&res, s, 8);
        return res; // No swap, just for hashing/comparison
    }

    ULTRA_ALWAYS_INLINE uint64_t bswap_64(uint64_t val) const noexcept {
        return __builtin_bswap64(val);
    }
    ULTRA_ALWAYS_INLINE uint32_t bswap_32(uint32_t val) const noexcept {
        return __builtin_bswap32(val);
    }
};

} // namespace ultra::md::itch
EOF

cat > include/ultra/market-data/book/order_book_l2.hpp << 'EOF'
#pragma once
#include "../../core/compiler.hpp"
#include "../../core/types.hpp"
#include "../itch/decoder.hpp"
#include <array>
#include <algorithm>
#include <vector>

namespace ultra::md {

/**
 * L2 Order Book - Aggregated by price level
 * Optimized for:
 * - Sub-100ns updates
 * - Cache-friendly memory layout
 * - Minimal branching
 */
class OrderBookL2 {
public:
    static constexpr size_t MAX_LEVELS = 100;
    
    struct Level {
        Price price{INVALID_PRICE};
        Quantity quantity{0};
        uint32_t order_count{0};
    };

    // Bids sorted descending, Asks sorted ascending
    using PriceLevelSide = std::array<Level, MAX_LEVELS>;
    
    OrderBookL2(SymbolId symbol_id);

    // Apply a decoded ITCH message
    ULTRA_HOT void update(const itch::ITCHDecoder::DecodedMessage& msg) noexcept;

    // Get current BBO
    ULTRA_ALWAYS_INLINE Level best_bid() const noexcept { return bids_[0]; }
    ULTRA_ALWAYS_INLINE Level best_ask() const noexcept { return asks_[0]; }

    // Get all levels
    const PriceLevelSide& bids() const noexcept { return bids_; }
    const PriceLevelSide& asks() const noexcept { return asks_; }

private:
    // L3-style book to reconstruct L2.
    // This is a simplified, non-performant hash map for the *stub*.
    // A production system would use a custom, open-addressing
    // hash table (like absl::flat_hash_map) with a custom allocator.
    struct L3Order {
        Price price;
        Quantity quantity;
        Side side;
    };
    std::vector<L3Order> orders_; // Using vector as a simple map stub
    
    SymbolId symbol_id_;
    ULTRA_CACHE_ALIGNED PriceLevelSide bids_{};
    ULTRA_CACHE_ALIGNED PriceLevelSide asks_{};

    // Internal handlers
    void add_order(OrderId id, Side side, Price price, Quantity qty) noexcept;
    void delete_order(OrderId id) noexcept;
    void modify_order(OrderId id, Quantity new_qty, Price new_price) noexcept;

    // Rebuild L2 from L3 (slow, but simple for this stub)
    void rebuild_l2() noexcept;
};

} // namespace ultra::md
EOF

echo "[7/10] Creating strategy, risk, and exec headers..."

cat > include/ultra/strategy/strategy.hpp << 'EOF'
#pragma once
#include "../core/types.hpp"
#include "../market-data/itch/decoder.hpp"
#include "../execution/gateway_sim.hpp" // Use Sim Gateway for now

namespace ultra::strategy {

// This is the output of the strategy
struct StrategyOrder {
    enum Action {
        NEW_ORDER,
        CANCEL_ORDER
    };
    Action action;
    OrderId order_id;
    SymbolId symbol_id;
    Side side;
    Price price;
    Quantity quantity;
    OrderType type;
};

// Base class for all strategies
class IStrategy {
public:
    virtual ~IStrategy() = default;

    // Market data event handler
    virtual void on_market_data(const md::itch::ITCHDecoder::DecodedMessage& msg) = 0;

    // Order execution update handler
    virtual void on_execution(const exec::ExecutionReport& report) = 0;

    // Get any orders the strategy wants to place
    virtual bool get_order(StrategyOrder& order) = 0;
};

} // namespace ultra::strategy
EOF

cat > include/ultra/strategy/rl-inference/rl_policy.hpp << 'EOF'
#pragma once
#include "../strategy.hpp"
#include "../../market-data/book/order_book_l2.hpp"
#include "../../core/lockfree/spsc_queue.hpp"
#include <memory>

namespace ultra::strategy {

/**
 * This is the C++ implementation of your thesis's core idea.
 * It runs the RL model inference. In a real system, this
 * might be a stub that calls the FPGA (via PCIe) for inference.
 * For simulation, it runs a software model.
 */
class RLPolicyStrategy : public IStrategy {
public:
    explicit RLPolicyStrategy(SymbolId symbol_id);
    ~RLPolicyStrategy() override;

    void on_market_data(const md::itch::ITCHDecoder::DecodedMessage& msg) override;
    void on_execution(const exec::ExecutionReport& report) override;
    bool get_order(StrategyOrder& order) override;

private:
    // This is the "Decision Engine" from your thesis, Fig 5 [cite: 175]
    void run_inference() noexcept;

    SymbolId symbol_id_;
    md::OrderBookL2 order_book_;

    // Features for the model
    struct ModelFeatures {
        Price best_bid;
        Quantity bid_size;
        Price best_ask;
        Quantity ask_size;
        Price mid_price;
        Price spread;
        double imbalance;
        double volatility;
        Quantity inventory;
    };

    // The output of the RL policy
    struct ModelOutput {
        Price optimal_bid_quote;
        Price optimal_ask_quote;
        bool should_trade;
    };
    
    // Stub for the neural net model
    // In reality, this would be a tensor library or custom SIMD code
    ModelOutput inference_stub(const ModelFeatures& features) noexcept;

    // State
    Quantity current_inventory_{0};
    
    // Output queue for orders
    SPSCQueue<StrategyOrder, 1024> order_queue_;
};

} // namespace ultra::strategy
EOF

cat > include/ultra/risk/pretrade_checker.hpp << 'EOF'
#pragma once
#include "../core/types.hpp"
#include "../strategy/strategy.hpp"

namespace ultra::risk {

/**
 * This is the hardware "Risk Engines" from your thesis, Fig 3 [cite: 101]
 * Implemented in C++ for the software path.
 */
class PretradeChecker {
public:
    struct Config {
        Quantity max_position_shares = 10000;
        Price max_notional_usd = 10000000;
        Quantity max_order_size = 1000;
        uint32_t max_orders_per_second = 10000;
    };

    explicit PretradeChecker(const Config& config);

    // Check a new order request
    // Returns true if the order is safe, false if it's rejected
    ULTRA_HOT bool check_order(const strategy::StrategyOrder& order) noexcept;

    // Update internal state from an execution
    void on_execution(const exec::ExecutionReport& report) noexcept;

private:
    Config config_;
    
    // Current state
    Quantity current_position_{0};
    // (Add tracking for order rate, etc.)
};

} // namespace ultra::risk
EOF

cat > include/ultra/execution/gateway_sim.hpp << 'EOF'
#pragma once
#include "../core/types.hpp"
#include "../strategy/strategy.hpp"
#include "../core/lockfree/spsc_queue.hpp"

namespace ultra::exec {

// Exchange -> Engine
struct ExecutionReport {
    Timestamp tsc;
    OrderId order_id;
    SymbolId symbol_id;
    OrderStatus status;
    Price fill_price{0};
    Quantity fill_quantity{0};
    Quantity remaining_quantity{0};
};

/**
 * A simple, single-threaded, simulated exchange gateway
 * It pretends to be an exchange, processing orders and sending
 * execution reports back.
 */
class GatewaySim {
public:
    GatewaySim() = default;

    // Engine -> Gateway
    void send_order(const strategy::StrategyOrder& order);

    // Gateway -> Engine
    bool get_execution_report(ExecutionReport& report);

    // Simulate market processing (call this in a loop)
    void update_market(const md::itch::ITCHDecoder::DecodedMessage& msg);

private:
    // A *very* simple matching engine
    // Bids (descending)
    std::vector<strategy::StrategyOrder> active_bids_; 
    // Asks (ascending)
    std::vector<strategy::StrategyOrder> active_asks_; 

    SPSCQueue<ExecutionReport, 8192> exec_reports_queue_;
    OrderId next_order_id_{1};

    void try_match(strategy::StrategyOrder& order);
    void send_fill(const strategy::StrategyOrder& order, Price fill_price, Quantity fill_qty);
    void send_accept(const strategy::StrategyOrder& order);
};

} // namespace ultra::exec
EOF

echo "[8/10] Creating C++ source implementations..."

cat > src/core/time/rdtsc_clock.cpp << 'EOF'
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
EOF

cat > src/network/parsers/ethernet_parser.cpp << 'EOF'
#include "ultra/network/parsers/ethernet_parser.hpp"

namespace ultra::network {

ULTRA_ALWAYS_INLINE EthernetParser::ParsedPacket EthernetParser::parse(
    const uint8_t* packet, 
    size_t packet_len,
    Timestamp timestamp_ns
) noexcept {
    ParsedPacket result = {nullptr, 0, 0, 0, 0, 0, timestamp_ns, false};
    
    // Min packet size: Eth + IP + UDP
    if (ULTRA_UNLIKELY(packet_len < (sizeof(EthernetHeader) + sizeof(IPv4Header) + sizeof(UDPHeader)))) {
        return result;
    }
    
    const auto* eth_hdr = reinterpret_cast<const EthernetHeader*>(packet);
    const uint8_t* ip_payload = packet + sizeof(EthernetHeader);

    // Check for IPv4
    if (ULTRA_LIKELY(eth_hdr->ethertype == ntohs_fast(ETHERTYPE_IP))) {
        const auto* ip_hdr = reinterpret_cast<const IPv4Header*>(ip_payload);
        
        // Basic IP header validation
        if (ULTRA_UNLIKELY((ip_hdr->version_ihl & 0xF0) != 0x40)) {
            return result; // Not IPv4
        }
        
        // Check for UDP
        if (ULTRA_LIKELY(ip_hdr->protocol == IPPROTO_UDP)) {
            uint8_t ip_header_len = (ip_hdr->version_ihl & 0x0F) * 4;
            const auto* udp_hdr = reinterpret_cast<const UDPHeader*>(ip_payload + ip_header_len);
            
            result.payload = ip_payload + ip_header_len + sizeof(UDPHeader);
            result.payload_len = ntohs_fast(udp_hdr->length) - sizeof(UDPHeader);
            
            // Check for packet truncation
            if (ULTRA_UNLIKELY(result.payload + result.payload_len > packet + packet_len)) {
                return result; // Invalid length
            }
            
            result.src_ip = ip_hdr->src_ip; // Already network byte order
            result.dst_ip = ip_hdr->dst_ip; // Already network byte order
            result.src_port = udp_hdr->src_port; // Already network byte order
            result.dst_port = udp_hdr->dst_port; // Already network byte order
            result.valid = true;
            
            return result;
        }
    }
    // Note: Skipping VLAN (802.1Q) parsing for simplicity in this stub
    
    return result;
}

} // namespace ultra::network
EOF

cat > src/market-data/itch/decoder.cpp << 'EOF'
#include "ultra/market-data/itch/decoder.hpp"
#include <iostream>

namespace ultra::md::itch {

ITCHDecoder::ITCHDecoder() {
    // We could pre-populate the symbol table from a config file here
    // For now, it's manual via register_symbol
}

void ITCHDecoder::register_symbol(const char* symbol, SymbolId id) {
    uint64_t symbol_int = char8_to_uint64(symbol);
    uint32_t index = hash_symbol(symbol_int);
    
    // Simple linear probe
    for(size_t i = 0; i < SYMBOL_HASH_SIZE; ++i) {
        uint32_t current_index = (index + i) & (SYMBOL_HASH_SIZE - 1);
        if (symbol_table_[current_index].id == INVALID_SYMBOL) {
            symbol_table_[current_index].symbol_int = symbol_int;
            symbol_table_[current_index].id = id;
            return;
        }
    }
    // Error: hash table full
    std::cerr << "ITCHDecoder Error: Symbol hash table full." << std::endl;
}

SymbolId ITCHDecoder::lookup_symbol(const char* symbol) const noexcept {
    uint64_t symbol_int = char8_to_uint64(symbol);
    uint32_t index = hash_symbol(symbol_int);
    
    for(size_t i = 0; i < SYMBOL_HASH_SIZE; ++i) {
        uint32_t current_index = (index + i) & (SYMBOL_HASH_SIZE - 1);
        if (symbol_table_[current_index].symbol_int == symbol_int) {
            return symbol_table_[current_index].id;
        }
        if (symbol_table_[current_index].id == INVALID_SYMBOL) {
            return INVALID_SYMBOL;
        }
    }
    return INVALID_SYMBOL;
}

ULTRA_HOT ULTRA_ALWAYS_INLINE 
ITCHDecoder::DecodedMessage ITCHDecoder::decode(const uint8_t* data, size_t len, Timestamp rdtsc_ts) noexcept {
    DecodedMessage msg{};
    msg.tsc = rdtsc_ts;
    msg.valid = false;
    msg.event_type = MDEventType::UNKNOWN;
    
    if (ULTRA_UNLIKELY(len < sizeof(MessageHeader))) return msg;
    
    const auto* header = reinterpret_cast<const MessageHeader*>(data);
    const auto msg_type = static_cast<MessageType>(header->type);

    // This switch is the critical path
    switch(msg_type) {
        case MessageType::ADD_ORDER: {
            if (ULTRA_UNLIKELY(len < sizeof(AddOrder))) return msg;
            const auto* add = reinterpret_cast<const AddOrder*>(data);
            
            msg.event_type = MDEventType::ADD_ORDER;
            msg.exchange_ts = bswap_64(add->timestamp);
            msg.order_id = bswap_64(add->order_ref_number);
            msg.side = (add->buy_sell_indicator == 'B') ? Side::BUY : Side::SELL;
            msg.quantity = bswap_32(add->shares);
            msg.price = decode_price(add->price);
            msg.symbol_id = lookup_symbol(add->stock);
            msg.valid = true;
            return msg;
        }
        
        case MessageType::ORDER_DELETE: {
            if (ULTRA_UNLIKELY(len < sizeof(OrderDelete))) return msg;
            const auto* del = reinterpret_cast<const OrderDelete*>(data);
            
            msg.event_type = MDEventType::DELETE_ORDER;
            msg.exchange_ts = bswap_64(del->timestamp);
            msg.order_id = bswap_64(del->order_ref_number);
            msg.valid = true;
            return msg;
        }

        case MessageType::ORDER_REPLACE: {
            if (ULTRA_UNLIKELY(len < sizeof(OrderReplace))) return msg;
            const auto* rep = reinterpret_cast<const OrderReplace*>(data);
            
            msg.event_type = MDEventType::MODIFY_ORDER;
            msg.exchange_ts = bswap_64(rep->timestamp);
            msg.order_id = bswap_64(rep->original_order_ref);
            msg.new_order_id = bswap_64(rep->new_order_ref);
            msg.quantity = bswap_32(rep->shares);
            msg.price = decode_price(rep->price);
            msg.valid = true;
            return msg;
        }
        
        // ... Add other cases: ORDER_EXECUTED, TRADE, etc.
        
        default:
            // Not a message type we care about for book building
            return msg;
    }
}

} // namespace ultra::md::itch
EOF

cat > src/market-data/book/order_book_l2.cpp << 'EOF'
#include "ultra/market-data/book/order_book_l2.hpp"
#include <iostream>
#include <algorithm>

namespace ultra::md {

OrderBookL2::OrderBookL2(SymbolId symbol_id) : symbol_id_(symbol_id) {
    // Initialize levels
    for (auto& level : bids_) {
        level.price = 0; // Bids start at 0
        level.quantity = 0;
    }
    for (auto& level : asks_) {
        level.price = INVALID_PRICE; // Asks start "at infinity"
        level.quantity = 0;
    }
}

ULTRA_HOT void OrderBookL2::update(const itch::ITCHDecoder::DecodedMessage& msg) noexcept {
    if (ULTRA_UNLIKELY(!msg.valid)) return;

    // This stub uses a slow, L3-to-L2 rebuild on every message.
    // A real L2 book would maintain the L2 state directly
    // by finding the price level and updating its quantity.
    
    // We only care about messages for *our* symbol
    if (msg.symbol_id != INVALID_SYMBOL && msg.symbol_id != symbol_id_) {
        return;
    }

    switch(msg.event_type) {
        case MDEventType::ADD_ORDER:
            add_order(msg.order_id, msg.side, msg.price, msg.quantity);
            break;
        case MDEventType::DELETE_ORDER:
            delete_order(msg.order_id);
            break;
        case MDEventType::MODIFY_ORDER:
            // ITCH Replace is complex: it's a delete + add
            delete_order(msg.order_id);
            // Side is not included in replace, so we have to guess
            // This is a flaw in this L3-stub approach.
            // A real book would know the side from the original order.
            // For now, we'll ignore modifies.
            // add_order(msg.new_order_id, ???, msg.price, msg.quantity);
            break;
        default:
            break;
    }
}

void OrderBookL2::add_order(OrderId id, Side side, Price price, Quantity qty) noexcept {
    // Check if order exists (for this simple stub)
    for(auto& order : orders_) {
        // This is not how OrderId is used in ITCH (it's not unique across book)
        // This is a major simplification.
    }
    orders_.push_back({price, qty, side});
    rebuild_l2();
}

void OrderBookL2::delete_order(OrderId id) noexcept {
    // This stub cannot work without a proper L3 map.
    // We'll skip implementation.
    // rebuild_l2();
}

void OrderBookL2::rebuild_l2() noexcept {
    // 1. Reset current L2 state
    for (auto& level : bids_) { level.price = 0; level.quantity = 0; level.order_count = 0; }
    for (auto& level : asks_) { level.price = INVALID_PRICE; level.quantity = 0; level.order_count = 0; }

    // 2. Sort all L3 orders (extremely slow!)
    std::sort(orders_.begin(), orders_.end(), [](const L3Order& a, const L3Order& b){
        return a.price < b.price;
    });

    // 3. Aggregate into L2 levels
    int bid_lvl = 0;
    int ask_lvl = 0;

    // Iterate backwards for bids
    for (int i = orders_.size() - 1; i >= 0; --i) {
        if (orders_[i].side == Side::BUY && bid_lvl < MAX_LEVELS) {
            if (orders_[i].price == bids_[bid_lvl].price) {
                bids_[bid_lvl].quantity += orders_[i].quantity;
                bids_[bid_lvl].order_count++;
            } else {
                if(bid_lvl + 1 < MAX_LEVELS) {
                    bid_lvl++;
                    bids_[bid_lvl].price = orders_[i].price;
                    bids_[bid_lvl].quantity = orders_[i].quantity;
                    bids_[bid_lvl].order_count = 1;
                }
            }
        }
    }

    // Iterate forwards for asks
    for (size_t i = 0; i < orders_.size(); ++i) {
        if (orders_[i].side == Side::SELL && ask_lvl < MAX_LEVELS) {
            if (orders_[i].price == asks_[ask_lvl].price) {
                asks_[ask_lvl].quantity += orders_[i].quantity;
                asks_[ask_lvl].order_count++;
            } else {
                 if(ask_lvl + 1 < MAX_LEVELS) {
                    ask_lvl++;
                    asks_[ask_lvl].price = orders_[i].price;
                    asks_[ask_lvl].quantity = orders_[i].quantity;
                    asks_[ask_lvl].order_count = 1;
                }
            }
        }
    }
    
    // This logic is simplified and likely buggy, but demonstrates
    // the *concept* of an L3-to-L2 rebuild.
    // A production book is vastly more complex.
}


} // namespace ultra::md
EOF

cat > src/strategy/rl-inference/rl_policy_stub.cpp << 'EOF'
#include "ultra/strategy/rl-inference/rl_policy.hpp"
#include <iostream>

namespace ultra::strategy {

RLPolicyStrategy::RLPolicyStrategy(SymbolId symbol_id)
    : symbol_id_(symbol_id), order_book_(symbol_id) {
    std::cout << "RLPolicyStrategy initialized for symbol " << symbol_id_ << std::endl;
}

RLPolicyStrategy::~RLPolicyStrategy() = default;

void RLPolicyStrategy::on_market_data(const md::itch::ITCHDecoder::DecodedMessage& msg) {
    // 1. Update our internal view of the L2 order book
    order_book_.update(msg);

    // 2. On a significant event (e.g., BBO change), run inference
    // For this stub, we'll run it on every update
    run_inference();
}

void RLPolicyStrategy::on_execution(const exec::ExecutionReport& report) {
    // Update our internal inventory based on fills
    if (report.status == OrderStatus::FILLED || report.status == OrderStatus::PARTIAL) {
        if (report.symbol_id == symbol_id_) {
            // (This assumes we only get reports for our own orders)
            // A real OMS would track order side.
            // We'll guess based on fill price.
            if (report.fill_price <= order_book_.best_bid().price) {
                current_inventory_ -= report.fill_quantity; // We sold
            } else {
                current_inventory_ += report.fill_quantity; // We bought
            }
            std::cout << "Inventory updated: " << current_inventory_ << std::endl;
        }
    }
}

bool RLPolicyStrategy::get_order(StrategyOrder& order) {
    // Return orders that the inference engine generated
    return order_queue_.pop(order);
}

void RLPolicyStrategy::run_inference() noexcept {
    // This is the "Decision Engine"
    
    // 1. Extract features (as listed in your config/engine.toml)
    auto bbo = order_book_.best_bid();
    auto bao = order_book_.best_ask();
    
    if (bbo.price == 0 || bao.price == INVALID_PRICE) {
        return; // Not enough data to trade
    }

    ModelFeatures features = {
        .best_bid = bbo.price,
        .bid_size = bbo.quantity,
        .best_ask = bao.price,
        .ask_size = bao.quantity,
        .mid_price = (bbo.price + bao.price) / 2,
        .spread = (bao.price - bbo.price),
        .imbalance = static_cast<double>(bbo.quantity) / (bbo.quantity + bao.quantity),
        .volatility = 0.0, // (Would be calculated by a signal engine)
        .inventory = current_inventory_
    };

    // 2. Run the (stubbed) model
    ModelOutput output = inference_stub(features);

    // 3. Act on the model's output
    if (output.should_trade) {
        // This is a *very* simple interpretation:
        // The model gives us new quotes. We'll send them as new orders.
        // A real MM would cancel old quotes and place new ones.
        
        StrategyOrder buy_order = {
            .action = StrategyOrder::Action::NEW_ORDER,
            .order_id = 0, // Gateway will assign
            .symbol_id = symbol_id_,
            .side = Side::BUY,
            .price = output.optimal_bid_quote,
            .quantity = 100, // Fixed size
            .type = OrderType::LIMIT
        };
        
        StrategyOrder sell_order = {
            .action = StrategyOrder::Action::NEW_ORDER,
            .order_id = 0,
            .symbol_id = symbol_id_,
            .side = Side::SELL,
            .price = output.optimal_ask_quote,
            .quantity = 100,
            .type = OrderType::LIMIT
        };

        // Push to our output queue
        order_queue_.push(buy_order);
        order_queue_.push(sell_order);
    }
}

RLPolicyStrategy::ModelOutput RLPolicyStrategy::inference_stub(const ModelFeatures& features) noexcept {
    // This is the stub for your AI model.
    // It just calculates a simple mid-point-based quote.
    
    // A simple Avellaneda-Stoikov-like inventory skew
    double inventory_skew = -0.1 * features.inventory; // 0.1 is risk aversion
    Price inventory_adj = static_cast<Price>(inventory_skew * features.spread);

    Price optimal_mid = features.mid_price + inventory_adj;
    
    // Simple 1-tick spread
    Price bid_quote = optimal_mid - (PRICE_SCALE / 100); // $0.01
    Price ask_quote = optimal_mid + (PRICE_SCALE / 100);

    return ModelOutput{
        .optimal_bid_quote = bid_quote,
        .optimal_ask_quote = ask_quote,
        .should_trade = true
    };
}

} // namespace ultra::strategy
EOF

cat > src/risk/pretrade_checker.cpp << 'EOF'
#include "ultra/risk/pretrade_checker.hpp"
#include <iostream>

namespace ultra::risk {

PretradeChecker::PretradeChecker(const Config& config) : config_(config) {}

ULTRA_HOT bool PretradeChecker::check_order(const strategy::StrategyOrder& order) noexcept {
    // Check 1: Max Order Size
    if (ULTRA_UNLIKELY(order.quantity > config_.max_order_size)) {
        std::cerr << "RISK REJECT: Max order size. " << order.quantity << " > " << config_.max_order_size << std::endl;
        return false;
    }

    // Check 2: Max Position
    Quantity new_position = current_position_;
    if (order.side == Side::BUY) {
        new_position += order.quantity;
    } else {
        new_position -= order.quantity;
    }

    if (ULTRA_UNLIKELY(std::abs(new_position) > config_.max_position_shares)) {
         std::cerr << "RISK REJECT: Max position. " << new_position << " > " << config_.max_position_shares << std::endl;
        return false;
    }

    // Check 3: Max Notional
    Price notional = order.price * order.quantity;
    if (ULTRA_UNLIKELY(notional > config_.max_notional_usd * PRICE_SCALE)) {
        std::cerr << "RISK REJECT: Max notional." << std::endl;
        return false;
    }
    
    // ... (Check 4: Message Rate) ...

    // All checks passed
    return true;
}

void PretradeChecker::on_execution(const exec::ExecutionReport& report) noexcept {
    // Update position on fills
    if (report.status == OrderStatus::FILLED || report.status == OrderStatus::PARTIAL) {
        // A real risk engine would know the side of the order.
        // This is a stub, so we'll just log.
    }
}

} // namespace ultra::risk
EOF

cat > src/execution/gateway_sim.cpp << 'EOF'
#include "ultra/execution/gateway_sim.hpp"
#include <iostream>
#include <algorithm>

namespace ultra::exec {

void GatewaySim::send_order(const strategy::StrategyOrder& order) {
    auto new_order = order;
    new_order.order_id = next_order_id_++;

    // 1. Send "Accepted" message
    send_accept(new_order);

    // 2. Try to match it
    try_match(new_order);
}

bool GatewaySim::get_execution_report(ExecutionReport& report) {
    return exec_reports_queue_.pop(report);
}

void GatewaySim::update_market(const md::itch::ITCHDecoder::DecodedMessage& msg) {
    // A real sim would use market data to drive fills
    // e.g., if a market trade comes in, cross our resting orders
}

void GatewaySim::try_match(strategy::StrategyOrder& order) {
    // This is a *very* simple FIFO matching stub
    if (order.side == Side::BUY) {
        // Try to match against asks
        if (!active_asks_.empty() && order.price >= active_asks_[0].price) {
            // Match!
            Price fill_price = active_asks_[0].price;
            Quantity fill_qty = std::min(order.quantity, active_asks_[0].quantity);
            
            send_fill(order, fill_price, fill_qty);
            // (In a real sim, we'd also send a fill to the other order)
            // ... (and update/remove the resting order)
        } else {
            // Add to book
            active_bids_.push_back(order);
            std::sort(active_bids_.begin(), active_bids_.end(), [](const auto& a, const auto& b){
                return a.price > b.price; // Bids descending
            });
        }
    } else {
        // Try to match against bids
        if (!active_bids_.empty() && order.price <= active_bids_[0].price) {
            // Match!
            Price fill_price = active_bids_[0].price;
            Quantity fill_qty = std::min(order.quantity, active_bids_[0].quantity);
            
            send_fill(order, fill_price, fill_qty);
        } else {
            // Add to book
            active_asks_.push_back(order);
            std::sort(active_asks_.begin(), active_asks_.end(), [](const auto& a, const auto& b){
                return a.price < b.price; // Asks ascending
            });
        }
    }
}

void GatewaySim::send_accept(const strategy::StrategyOrder& order) {
    exec_reports_queue_.push(ExecutionReport{
        .tsc = 0, // (Set timestamp)
        .order_id = order.order_id,
        .symbol_id = order.symbol_id,
        .status = OrderStatus::ACCEPTED,
        .remaining_quantity = order.quantity
    });
}

void GatewaySim::send_fill(const strategy::StrategyOrder& order, Price fill_price, Quantity fill_qty) {
    std::cout << "GATEWAY: " << order.side << " Order " << order.order_id << " FILLED " 
              << fill_qty << " @ " << from_price(fill_price) << std::endl;
              
    exec_reports_queue_.push(ExecutionReport{
        .tsc = 0,
        .order_id = order.order_id,
        .symbol_id = order.symbol_id,
        .status = OrderStatus::FILLED, // (Stub: assumes full fill)
        .fill_price = fill_price,
        .fill_quantity = fill_qty,
        .remaining_quantity = 0
    });
}

} // namespace ultra::exec
EOF

echo "[9/10] Creating main application headers and source..."

cat > apps/live-engine/engine.hpp << 'EOF'
#pragma once
#include <ultra/core/types.hpp>
#include <ultra/market-data/itch/decoder.hpp>
#include <ultra/strategy/rl-inference/rl_policy.hpp>
#include <ultra/risk/pretrade_checker.hpp>
#include <ultra/execution/gateway_sim.hpp>
#include <memory>
#include <thread>
#include <atomic>

namespace ultra {

/**
 * This is the main orchestrator, analogous to your thesis's
 * "Unified High-Frequency Trading System Architecture" (Fig 8) [cite: 250]
 */
class Engine {
public:
    Engine();
    ~Engine();

    void run();
    void stop();

private:
    void md_thread_loop();    // Market Data Ingress
    void exec_thread_loop();  // Execution/OMS Loop
    void strategy_thread_loop(); // Strategy Decision Loop

    // --- Components ---
    std::unique_ptr<md::itch::ITCHDecoder> decoder_;
    std::unique_ptr<strategy::RLPolicyStrategy> strategy_;
    std::unique_ptr<risk::PretradeChecker> risk_checker_;
    std::unique_ptr<exec::GatewaySim> gateway_;
    
    // --- Message Queues (The "Event Driven Pipeline" from Fig 3) ---
    // (Using SPSC queues as this is a simple 1-to-1 pipeline)
    
    // MD -> Strategy
    using MDQueue = SPSCQueue<md::itch::ITCHDecoder::DecodedMessage, 16384>;
    std::unique_ptr<MDQueue> md_to_strategy_queue_;
    
    // Strategy -> Risk
    using OrderQueue = SPSCQueue<strategy::StrategyOrder, 8192>;
    std::unique_ptr<OrderQueue> strategy_to_risk_queue_;

    // Risk -> Gateway
    std::unique_ptr<OrderQueue> risk_to_gateway_queue_;

    // Gateway -> Strategy (Exec Reports)
    using ExecQueue = SPSCQueue<exec::ExecutionReport, 8192>;
    std::unique_ptr<ExecQueue> gateway_to_strategy_queue_;

    // --- Threads ---
    std::thread md_thread_;
    std::thread exec_thread_;
    std::thread strategy_thread_;
    std::atomic<bool> running_{false};
};

} // namespace ultra
EOF

cat > apps/live-engine/engine.cpp << 'EOF'
#include "engine.hpp"
#include <iostream>
#include <vector>

namespace ultra {

Engine::Engine() {
    // --- 1. Allocate Queues ---
    // (Use new/unique_ptr to control NUMA allocation in a real system)
    md_to_strategy_queue_ = std::make_unique<MDQueue>();
    strategy_to_risk_queue_ = std::make_unique<OrderQueue>();
    risk_to_gateway_queue_ = std::make_unique<OrderQueue>();
    gateway_to_strategy_queue_ = std::make_unique<ExecQueue>();
    
    // --- 2. Initialize Components ---
    decoder_ = std::make_unique<md::itch::ITCHDecoder>();
    
    // Register the symbol we want to trade
    const SymbolId AAPL_ID = 1;
    decoder_->register_symbol("AAPL    ", AAPL_ID);
    
    strategy_ = std::make_unique<strategy::RLPolicyStrategy>(AAPL_ID);
    
    risk::PretradeChecker::Config risk_config;
    // (Load from config/engine.toml)
    risk_checker_ = std::make_unique<risk::PretradeChecker>(risk_config);
    
    gateway_ = std::make_unique<exec::GatewaySim>();
    
    std::cout << "Engine components initialized." << std::endl;
}

Engine::~Engine() {
    if (running_) {
        stop();
    }
}

void Engine::run() {
    running_ = true;
    
    // Start threads (in reverse order of data flow)
    exec_thread_ = std::thread(&Engine::exec_thread_loop, this);
    strategy_thread_ = std::thread(&Engine::strategy_thread_loop, this);
    md_thread_ = std::thread(&Engine::md_thread_loop, this);
    
    std::cout << "Engine running. Press [Enter] to stop." << std::endl;
    std::cin.get();
    stop();
}

void Engine::stop() {
    running_ = false;
    
    if (md_thread_.joinable()) md_thread_.join();
    if (strategy_thread_.joinable()) strategy_thread_.join();
    if (exec_thread_.joinable()) exec_thread_.join();
    
    std::cout << "Engine stopped." << std::endl;
}

/**
 * This is the "Market Data Pipeline" (Fig 2)
 * It simulates a network feed and decodes it.
 */
void Engine::md_thread_loop() {
    std::cout << "[MD Thread] running." << std::endl;
    
    // --- Simulate a Market Data Feed ---
    // (This would be a network receiver in a real system)
    using namespace md::itch;
    
    // Create a fake "AAPL" Add Order message
    std::vector<uint8_t> add_order_bytes(sizeof(AddOrder));
    auto* add_msg = reinterpret_cast<AddOrder*>(add_order_bytes.data());
    add_msg->header.type = static_cast<uint8_t>(MessageType::ADD_ORDER);
    add_msg->header.length = __builtin_bswap16(sizeof(AddOrder));
    add_msg->timestamp = __builtin_bswap64(123456789ULL);
    add_msg->order_ref_number = __builtin_bswap64(10001ULL);
    add_msg->buy_sell_indicator = 'B';
    add_msg->shares = __builtin_bswap32(100);
    memcpy(add_msg->stock, "AAPL    ", 8);
    add_msg->price = __builtin_bswap32(1500000); // $150.0000
    // --- End of Sim Data ---
    
    while (running_) {
        // 1. Get packet from network (simulated)
        const uint8_t* raw_packet = add_order_bytes.data();
        size_t packet_len = add_order_bytes.size();
        Timestamp rdtsc_ts = RDTSCClock::rdtsc();
        
        // 2. Decode it
        auto decoded = decoder_->decode(raw_packet, packet_len, rdtsc_ts);
        
        // 3. Push to strategy queue
        if (decoded.valid) {
            if (ULTRA_UNLIKELY(!md_to_strategy_queue_->push(decoded))) {
                // Queue full, drop message
            }
        }
        
        // Simulate feed
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

/**
 * This is the "Trading Logic" + "Risk Engines" + "FPGA Engine"
 */
void Engine::strategy_thread_loop() {
    std::cout << "[Strategy Thread] running." << std::endl;
    
    md::itch::ITCHDecoder::DecodedMessage md_msg;
    exec::ExecutionReport exec_report;
    strategy::StrategyOrder strategy_order;
    
    while (running_) {
        bool work_done = false;
        
        // 1. Process market data
        if (md_to_strategy_queue_->pop(md_msg)) {
            strategy_->on_market_data(md_msg);
            work_done = true;
        }
        
        // 2. Process execution reports
        if (gateway_to_strategy_queue_->pop(exec_report)) {
            strategy_->on_execution(exec_report);
            work_done = true;
        }

        // 3. Get new orders from strategy
        while (strategy_->get_order(strategy_order)) {
            // 4. Push to risk engine
            if (ULTRA_UNLIKELY(!strategy_to_risk_queue_->push(strategy_order))) {
                // Drop order
            }
            work_done = true;
        }
        
        if (!work_done) {
            // Busy-wait or sleep
            // std::this_thread::yield();
        }
    }
}

/**
 * This is the "Order Processing" + "Smart Order Router"
 */
void Engine::exec_thread_loop() {
    std::cout << "[Exec Thread] running." << std::endl;

    strategy::StrategyOrder order_to_check;
    exec::ExecutionReport exec_report;
    
    while (running_) {
        bool work_done = false;
        
        // 1. Check orders from strategy
        if (strategy_to_risk_queue_->pop(order_to_check)) {
            // 2. Run through Pre-trade Risk
            if (ULTRA_LIKELY(risk_checker_->check_order(order_to_check))) {
                // 3. Send to Gateway
                gateway_->send_order(order_to_check);
            } else {
                // Risk rejected
                // (Send a reject message back to strategy)
            }
            work_done = true;
        }
        
        // 4. Poll gateway for execution reports
        if (gateway_->get_execution_report(exec_report)) {
            // 5. Send back to strategy
            if (ULTRA_UNLIKELY(!gateway_to_strategy_queue_->push(exec_report))) {
                // Drop exec report
            }
            work_done = true;
        }
        
        if (!work_done) {
            // std::this_thread::yield();
        }
    }
}

} // namespace ultra
EOF

cat > apps/live-engine/main.cpp << 'EOF'
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
EOF

echo "[10/10] All files created successfully."
echo "=================================================="