#include "ultra/risk/pretrade_checker.hpp"
#include "ultra/execution/gateway_sim.hpp" // <-- FIX: Added full include
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
    // This code is now valid because we included the full definition
    if (report.status == OrderStatus::FILLED || report.status == OrderStatus::PARTIAL) {
        // A real risk engine would know the side of the order.
        // This is a stub, so we'll just log.
    }
}

} // namespace ultra::risk