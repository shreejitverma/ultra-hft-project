#include "ultra/strategy/rl-inference/rl_policy.hpp"
#include "ultra/execution/gateway_sim.hpp" // <-- ADDED FULL INCLUDE
#include <iostream>

namespace ultra::strategy {

RLPolicyStrategy::RLPolicyStrategy(SymbolId symbol_id)
    : symbol_id_(symbol_id), order_book_(symbol_id) {
    std::cout << "RLPolicyStrategy initialized for symbol " << symbol_id_ << std::endl;
}

RLPolicyStrategy::~RLPolicyStrategy() = default;

void RLPolicyStrategy::on_market_data(const md::itch::Decoder::DecodedMessage& msg) {
    // 1. Update our internal view of the L2 order book
    order_book_.update(msg);

    // 2. On a significant event (e.g., BBO change), run inference
    // For this stub, we'll run it on every update
    run_inference();
}

void RLPolicyStrategy::on_execution(const exec::ExecutionReport& report) {
    // Update our internal inventory based on fills
    if (report.status == OrderStatus::FILLED || report.status == OrderStatus::PARTIAL) {
        if (report.symbol_id == symbol_id_) {
            // (This assumes we only get reports for our own orders)
            // A real OMS would track order side.
            // We'll guess based on fill price.
            if (report.fill_price <= order_book_.best_bid().price) {
                current_inventory_ -= report.fill_quantity; // We sold
            } else {
                current_inventory_ += report.fill_quantity; // We bought
            }
            std::cout << "Inventory updated: " << current_inventory_ << std::endl;
        }
    }
}

bool RLPolicyStrategy::get_order(StrategyOrder& order) {
    // Return orders that the inference engine generated
    return order_queue_.pop(order);
}

void RLPolicyStrategy::run_inference() noexcept {
    // This is the "Decision Engine"
    
    // 1. Extract features (as listed in your config/engine.toml)
    auto bbo = order_book_.best_bid();
    auto bao = order_book_.best_ask();
    
    if (bbo.price == 0 || bao.price == INVALID_PRICE) {
        return; // Not enough data to trade
    }

    ModelFeatures features = {
        .best_bid = bbo.price,
        .bid_size = bbo.quantity,
        .best_ask = bao.price,
        .ask_size = bao.quantity,
        .mid_price = (bbo.price + bao.price) / 2,
        .spread = (bao.price - bbo.price),
        .imbalance = static_cast<double>(bbo.quantity) / (bbo.quantity + bao.quantity),
        .volatility = 0.0, // (Would be calculated by a signal engine)
        .inventory = current_inventory_
    };

    // 2. Run the (stubbed) model
    ModelOutput output = inference_stub(features);

    // 3. Act on the model's output
    if (output.should_trade) {
        // This is a *very* simple interpretation:
        // The model gives us new quotes. We'll send them as new orders.
        // A real MM would cancel old quotes and place new ones.
        
        StrategyOrder buy_order = {
            .action = StrategyOrder::Action::NEW_ORDER,
            .order_id = 0, // Gateway will assign
            .symbol_id = symbol_id_,
            .side = Side::BUY,
            .price = output.optimal_bid_quote,
            .quantity = 100, // Fixed size
            .type = OrderType::LIMIT
        };
        
        StrategyOrder sell_order = {
            .action = StrategyOrder::Action::NEW_ORDER,
            .order_id = 0,
            .symbol_id = symbol_id_,
            .side = Side::SELL,
            .price = output.optimal_ask_quote,
            .quantity = 100,
            .type = OrderType::LIMIT
        };

        // Push to our output queue
        order_queue_.push(buy_order);
        order_queue_.push(sell_order);
    }
}

RLPolicyStrategy::ModelOutput RLPolicyStrategy::inference_stub(const ModelFeatures& features) noexcept {
    // This is the stub for your AI model.
    // It just calculates a simple mid-point-based quote.
    
    // A simple Avellaneda-Stoikov-like inventory skew
    double inventory_skew = -0.1 * features.inventory; // 0.1 is risk aversion
    Price inventory_adj = static_cast<Price>(inventory_skew * features.spread);

    Price optimal_mid = features.mid_price + inventory_adj;
    
    // Simple 1-tick spread
    Price bid_quote = optimal_mid - (PRICE_SCALE / 100); // $0.01
    Price ask_quote = optimal_mid + (PRICE_SCALE / 100);

    return ModelOutput{
        .optimal_bid_quote = bid_quote,
        .optimal_ask_quote = ask_quote,
        .should_trade = true
    };
}

} // namespace ultra::strategy