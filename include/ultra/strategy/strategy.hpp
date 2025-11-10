#pragma once
#include "../core/types.hpp"
#include "../market-data/itch/decoder.hpp" // This line was correct

namespace ultra::exec {
    struct ExecutionReport; // Forward declaration
}

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
    // FIX: Changed "Decoder" to "ITCHDecoder"
    virtual void on_market_data(const md::itch::ITCHDecoder::DecodedMessage& msg) = 0;

    // Order execution update handler
    virtual void on_execution(const exec::ExecutionReport& report) = 0;

    // Get any orders the strategy wants to place
    virtual bool get_order(StrategyOrder& order) = 0;
};

} // namespace ultra::strategy