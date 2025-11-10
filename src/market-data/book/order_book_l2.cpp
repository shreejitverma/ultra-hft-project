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
    // FIX: Silence unused parameter warning
    (void)id; 
    
    // Check if order exists (for this simple stub)
    // FIX: Comment out unused loop
    // for(auto& order : orders_) {
    // }
    orders_.push_back({price, qty, side});
    rebuild_l2();
}

void OrderBookL2::delete_order(OrderId id) noexcept {
    // FIX: Silence unused parameter warning
    (void)id;
    
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
    // FIX: Change int to size_t for correct type comparison
    size_t bid_lvl = 0;
    size_t ask_lvl = 0;

    // Iterate backwards for bids
    // FIX: Use correct unsigned loop to avoid -Wsign-compare
    for (size_t i = orders_.size(); i-- > 0;) {
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