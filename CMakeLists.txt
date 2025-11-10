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

# Telemetry library (FIX: Changed STATIC to INTERFACE and removed sources)
add_library(ultra_telemetry INTERFACE)
target_link_libraries(ultra_telemetry INTERFACE ultra_core)

# Combined library for applications (FIX: Changed STATIC to INTERFACE)
add_library(ultra_hft INTERFACE)
target_link_libraries(ultra_hft INTERFACE
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