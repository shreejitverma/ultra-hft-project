#pragma once
#include "compiler.hpp" // <-- THIS IS THE FIX
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