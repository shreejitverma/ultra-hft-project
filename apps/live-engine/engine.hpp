#pragma once
#include <ultra/core/types.hpp>
#include <ultra/market-data/itch/decoder.hpp>
#include <ultra/strategy/rl-inference/rl_policy.hpp>
#include <ultra/risk/pretrade_checker.hpp>
#include <ultra/execution/gateway_sim.hpp>
#include <memory>
#include <thread>
#include <atomic>

namespace ultra {

/**
 * This is the main orchestrator, analogous to your thesis's
 * "Unified High-Frequency Trading System Architecture" (Fig 8) [cite: 250]
 */
class Engine {
public:
    Engine();
    ~Engine();

    void run();
    void stop();

private:
    void md_thread_loop();    // Market Data Ingress
    void exec_thread_loop();  // Execution/OMS Loop
    void strategy_thread_loop(); // Strategy Decision Loop

    // --- Components ---
    std::unique_ptr<md::itch::ITCHDecoder> decoder_;
    std::unique_ptr<strategy::RLPolicyStrategy> strategy_;
    std::unique_ptr<risk::PretradeChecker> risk_checker_;
    std::unique_ptr<exec::GatewaySim> gateway_;
    
    // --- Message Queues (The "Event Driven Pipeline" from Fig 3) ---
    // (Using SPSC queues as this is a simple 1-to-1 pipeline)
    
    // MD -> Strategy
    using MDQueue = SPSCQueue<md::itch::ITCHDecoder::DecodedMessage, 16384>;
    std::unique_ptr<MDQueue> md_to_strategy_queue_;
    
    // Strategy -> Risk
    using OrderQueue = SPSCQueue<strategy::StrategyOrder, 8192>;
    std::unique_ptr<OrderQueue> strategy_to_risk_queue_;

    // Risk -> Gateway
    std::unique_ptr<OrderQueue> risk_to_gateway_queue_;

    // Gateway -> Strategy (Exec Reports)
    using ExecQueue = SPSCQueue<exec::ExecutionReport, 8192>;
    std::unique_ptr<ExecQueue> gateway_to_strategy_queue_;

    // --- Threads ---
    std::thread md_thread_;
    std::thread exec_thread_;
    std::thread strategy_thread_;
    std::atomic<bool> running_{false};
};

} // namespace ultra
