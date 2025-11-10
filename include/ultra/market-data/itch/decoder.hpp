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
