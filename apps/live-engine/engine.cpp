#include "engine.hpp"
#include <iostream>
#include <vector>

namespace ultra {

Engine::Engine() {
    // --- 1. Allocate Queues ---
    // (Use new/unique_ptr to control NUMA allocation in a real system)
    md_to_strategy_queue_ = std::make_unique<MDQueue>();
    strategy_to_risk_queue_ = std::make_unique<OrderQueue>();
    risk_to_gateway_queue_ = std::make_unique<OrderQueue>();
    gateway_to_strategy_queue_ = std::make_unique<ExecQueue>();
    
    // --- 2. Initialize Components ---
    decoder_ = std::make_unique<md::itch::ITCHDecoder>();
    
    // Register the symbol we want to trade
    const SymbolId AAPL_ID = 1;
    decoder_->register_symbol("AAPL    ", AAPL_ID);
    
    strategy_ = std::make_unique<strategy::RLPolicyStrategy>(AAPL_ID);
    
    risk::PretradeChecker::Config risk_config;
    // (Load from config/engine.toml)
    risk_checker_ = std::make_unique<risk::PretradeChecker>(risk_config);
    
    gateway_ = std::make_unique<exec::GatewaySim>();
    
    std::cout << "Engine components initialized." << std::endl;
}

Engine::~Engine() {
    if (running_) {
        stop();
    }
}

void Engine::run() {
    running_ = true;
    
    // Start threads (in reverse order of data flow)
    exec_thread_ = std::thread(&Engine::exec_thread_loop, this);
    strategy_thread_ = std::thread(&Engine::strategy_thread_loop, this);
    md_thread_ = std::thread(&Engine::md_thread_loop, this);
    
    std::cout << "Engine running. Press [Enter] to stop." << std::endl;
    std::cin.get();
    stop();
}

void Engine::stop() {
    running_ = false;
    
    if (md_thread_.joinable()) md_thread_.join();
    if (strategy_thread_.joinable()) strategy_thread_.join();
    if (exec_thread_.joinable()) exec_thread_.join();
    
    std::cout << "Engine stopped." << std::endl;
}

/**
 * This is the "Market Data Pipeline" (Fig 2)
 * It simulates a network feed and decodes it.
 */
void Engine::md_thread_loop() {
    std::cout << "[MD Thread] running." << std::endl;
    
    // --- Simulate a Market Data Feed ---
    // (This would be a network receiver in a real system)
    using namespace md::itch;
    
    // Create a fake "AAPL" Add Order message
    std::vector<uint8_t> add_order_bytes(sizeof(AddOrder));
    auto* add_msg = reinterpret_cast<AddOrder*>(add_order_bytes.data());
    add_msg->header.type = static_cast<uint8_t>(MessageType::ADD_ORDER);
    add_msg->header.length = __builtin_bswap16(sizeof(AddOrder));
    add_msg->timestamp = __builtin_bswap64(123456789ULL);
    add_msg->order_ref_number = __builtin_bswap64(10001ULL);
    add_msg->buy_sell_indicator = 'B';
    add_msg->shares = __builtin_bswap32(100);
    memcpy(add_msg->stock, "AAPL    ", 8);
    add_msg->price = __builtin_bswap32(1500000); // $150.0000
    // --- End of Sim Data ---
    
    while (running_) {
        // 1. Get packet from network (simulated)
        const uint8_t* raw_packet = add_order_bytes.data();
        size_t packet_len = add_order_bytes.size();
        Timestamp rdtsc_ts = RDTSCClock::rdtsc();
        
        // 2. Decode it
        auto decoded = decoder_->decode(raw_packet, packet_len, rdtsc_ts);
        
        // 3. Push to strategy queue
        if (decoded.valid) {
            if (ULTRA_UNLIKELY(!md_to_strategy_queue_->push(decoded))) {
                // Queue full, drop message
            }
        }
        
        // Simulate feed
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

/**
 * This is the "Trading Logic" + "Risk Engines" + "FPGA Engine"
 */
void Engine::strategy_thread_loop() {
    std::cout << "[Strategy Thread] running." << std::endl;
    
    md::itch::ITCHDecoder::DecodedMessage md_msg;
    exec::ExecutionReport exec_report;
    strategy::StrategyOrder strategy_order;
    
    while (running_) {
        bool work_done = false;
        
        // 1. Process market data
        if (md_to_strategy_queue_->pop(md_msg)) {
            strategy_->on_market_data(md_msg);
            work_done = true;
        }
        
        // 2. Process execution reports
        if (gateway_to_strategy_queue_->pop(exec_report)) {
            strategy_->on_execution(exec_report);
            work_done = true;
        }

        // 3. Get new orders from strategy
        while (strategy_->get_order(strategy_order)) {
            // 4. Push to risk engine
            if (ULTRA_UNLIKELY(!strategy_to_risk_queue_->push(strategy_order))) {
                // Drop order
            }
            work_done = true;
        }
        
        if (!work_done) {
            // Busy-wait or sleep
            // std::this_thread::yield();
        }
    }
}

/**
 * This is the "Order Processing" + "Smart Order Router"
 */
void Engine::exec_thread_loop() {
    std::cout << "[Exec Thread] running." << std::endl;

    strategy::StrategyOrder order_to_check;
    exec::ExecutionReport exec_report;
    
    while (running_) {
        bool work_done = false;
        
        // 1. Check orders from strategy
        if (strategy_to_risk_queue_->pop(order_to_check)) {
            // 2. Run through Pre-trade Risk
            if (ULTRA_LIKELY(risk_checker_->check_order(order_to_check))) {
                // 3. Send to Gateway
                gateway_->send_order(order_to_check);
            } else {
                // Risk rejected
                // (Send a reject message back to strategy)
            }
            work_done = true;
        }
        
        // 4. Poll gateway for execution reports
        if (gateway_->get_execution_report(exec_report)) {
            // 5. Send back to strategy
            if (ULTRA_UNLIKELY(!gateway_to_strategy_queue_->push(exec_report))) {
                // Drop exec report
            }
            work_done = true;
        }
        
        if (!work_done) {
            // std::this_thread::yield();
        }
    }
}

} // namespace ultra
