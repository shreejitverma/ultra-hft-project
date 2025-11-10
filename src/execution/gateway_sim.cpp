#include "ultra/execution/gateway_sim.hpp"
#include "ultra/strategy/strategy.hpp" // <-- ADDED FULL INCLUDE
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