#pragma once
#include "../strategy.hpp"
#include "../../market-data/book/order_book_l2.hpp"
#include "../../core/lockfree/spsc_queue.hpp"
#include <memory>

namespace ultra::strategy {

/**
 * This is the C++ implementation of your thesis's core idea.
 * It runs the RL model inference. In a real system, this
 * might be a stub that calls the FPGA (via PCIe) for inference.
 * For simulation, it runs a software model.
 */
class RLPolicyStrategy : public IStrategy {
public:
    explicit RLPolicyStrategy(SymbolId symbol_id);
    ~RLPolicyStrategy() override;

    void on_market_data(const md::itch::ITCHDecoder::DecodedMessage& msg) override;
    void on_execution(const exec::ExecutionReport& report) override;
    bool get_order(StrategyOrder& order) override;

private:
    // This is the "Decision Engine" from your thesis, Fig 5 [cite: 175]
    void run_inference() noexcept;

    SymbolId symbol_id_;
    md::OrderBookL2 order_book_;

    // Features for the model
    struct ModelFeatures {
        Price best_bid;
        Quantity bid_size;
        Price best_ask;
        Quantity ask_size;
        Price mid_price;
        Price spread;
        double imbalance;
        double volatility;
        Quantity inventory;
    };

    // The output of the RL policy
    struct ModelOutput {
        Price optimal_bid_quote;
        Price optimal_ask_quote;
        bool should_trade;
    };
    
    // Stub for the neural net model
    // In reality, this would be a tensor library or custom SIMD code
    ModelOutput inference_stub(const ModelFeatures& features) noexcept;

    // State
    Quantity current_inventory_{0};
    
    // Output queue for orders
    SPSCQueue<StrategyOrder, 1024> order_queue_;
};

} // namespace ultra::strategy
