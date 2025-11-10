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
