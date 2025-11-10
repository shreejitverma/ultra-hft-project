#pragma once
#include "../core/types.hpp"
#include "../core/lockfree/spsc_queue.hpp"
#include <vector>
// #include "../strategy/strategy.hpp" // <-- REMOVED THIS

namespace ultra::md::itch {
    class ITCHDecoder; // <-- NEEDED FOR DecodedMessage
}

namespace ultra::strategy {
    struct StrategyOrder; // <-- ADDED FORWARD DECLARATION
}

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