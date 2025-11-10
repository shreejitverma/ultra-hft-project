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
