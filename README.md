# Trishul: An AI-Driven Ultra-Low-Latency Market Making and Execution Platform
A fast, sharp, and lethal system to execute trades

This project is the **C++ software implementation** of the *Hybrid Control Plane* and simulation engine described in the Master's thesis, **"AI-Integrated FPGA for Market Making in Volatile Environments."**

It provides a **production-grade, ultra-low-latency C++ skeleton** for developing, backtesting, and deploying high-frequency market-making strategies. The architecture supports a **hybrid CPU/FPGA system**, where this C++ application serves as the *software path* for simulation, risk management, and AI model control on the FPGA.

---

## Key Features

### Ultra-Low Latency Design
The C++ code is built for **speed and determinism**, using:
- **RDTSC Clock** – High-resolution, sub-nanosecond timestamping (`rdtsc_clock.hpp`).
- **Lock-Free Queues** – Contention-free inter-thread communication (`spsc_queue.hpp`).
- **Compiler Optimizations** – `-march=native`, `-mavx2`, and other SIMD-friendly flags.
- **Cache-Friendly Layouts** – `ULTRA_CACHE_ALIGNED` macros to prevent false sharing.

### Event-Driven Pipeline
A multi-threaded, event-driven system where components communicate via SPSC queues, mirroring real HFT architecture.

### Hybrid Architecture Ready
The `RLPolicyStrategy` acts as the AI/RL model stub, ready for FPGA communication and inference integration.

### Production Parity
Built and run inside a `linux/amd64` Docker container (even on Apple Silicon) for full parity with production x86-64 servers.

---

## ⚙️ System Architecture: Event-Driven Pipeline

The live-engine is the **heart of the system**, orchestrating all threads and components.

```mermaid
graph TD
    subgraph "MD Thread (md_thread_loop)"
        style MD_Thread fill:#222,stroke:#333,color:#fff
        A[Simulated MD Feed] -->|Raw ITCH Packet| B[ITCHDecoder]
        B -->|DecodedMessage| C(md_to_strategy_queue)
    end

    subgraph "Strategy Thread (strategy_thread_loop)"
        style Strategy_Thread fill:#222,stroke:#333,color:#fff
        C --> D{Strategy Thread Loop}
        D -->|on_market_data| E[RLPolicyStrategy]
        E -->|StrategyOrder| F(strategy_to_risk_queue)
    end

    subgraph "Exec Thread (exec_thread_loop)"
        style Exec_Thread fill:#222,stroke:#333,color:#fff
        F --> G{Exec Thread Loop}
        G -->|check_order| H[PretradeChecker]
        H -->|Order OK| I(risk_to_gateway_queue)
        I --> G
        G -->|send_order| J[GatewaySim]
    end

    subgraph "Feedback Loop"
        style Feedback_Loop fill:#222,stroke:#333,color:#fff
        J -->|ExecutionReport| K(gateway_to_strategy_queue)
        K --> D
        D -->|on_execution| E
    end
````

---

## How to Build and Run

This project **must be built in an x86-64 Linux environment.**
A preconfigured **Dockerfile** handles this automatically.

### 1. Build the Environment

```bash
docker build -t ultra-hft-env .
```

### 2. Enter the Container

```bash
docker run -it --rm -v "$(pwd):/home/builder/project" ultra-hft-env
```

You’ll now be inside the container environment:

```
builder@container:~/project$
```

### 3. Build the Engine

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make live_engine -j
```

### 4. Run the Engine

```bash
./apps/live-engine/live_engine
```

You’ll see simulated gateway activity, order placements, and fills.

---

## Component & File Deep Dive

### `apps/live-engine/`

**main.cpp**

* Entry point.
* Calls `RDTSCClock::calibrate()` to establish nanosecond/tick conversion.
* Creates and runs the main `Engine`.

**engine.hpp**

* Defines the “brain” of the app.
* Declares `std::thread` members and atomic shutdown signals.
* Manages all major components (decoder, strategy, risk, gateway).

**engine.cpp**
Implements all thread loops:

1. **md_thread_loop()** – Simulates market data ingestion.
2. **strategy_thread_loop()** – Calls `on_market_data()` or `on_execution()`, generates new orders.
3. **exec_thread_loop()** – Runs risk checks, sends to gateway, handles execution feedback.

---

### `include/ultra/core/`

**compiler.hpp**

* Defines macros for performance:

  * `ULTRA_ALWAYS_INLINE`
  * `ULTRA_CACHE_ALIGNED`

**types.hpp**

* Defines all fundamental types (`Price`, `Quantity`, `OrderId`, etc.).
* Uses **fixed-point arithmetic** (no floating point).

**rdtsc_clock.hpp / .cpp**

* Implements sub-nanosecond precision timer.
* `calibrate()` computes conversion between CPU ticks and nanoseconds.
* `now()` returns precise timestamps.

**lockfree/spsc_queue.hpp**

* **Single Producer Single Consumer Queue** (lock-free).
* Uses atomic `head_` and `tail_` for thread-safe ring buffer.
* Cache-aligned for optimal throughput.

---

### `include/ultra/network/`

**ethernet_parser.hpp / .cpp**

* Zero-copy Ethernet/IP/UDP header parsing.
* Uses `reinterpret_cast` to overlay structs on raw memory buffers.

---

### `include/ultra/market-data/`

**itch/decoder.hpp / .cpp**

* Implements **L2 ITCH 5.0 protocol decoding**.
* Zero-copy decoder optimized via jump-table switch.
* Converts Big Endian to Little Endian with `__builtin_bswap`.

**book/order_book_l2.hpp / .cpp**

* Stub for Level 2 order book.
* Currently uses `std::vector` and full re-sorting (for clarity).
* To be replaced with cache-optimized implementation.

---

### `include/ultra/strategy/`

**strategy.hpp**

* Defines base interface `IStrategy` with:

  * `on_market_data()`
  * `on_execution()`
  * `get_order()`

**rl-inference/rl_policy.hpp / .cpp**

* C++ implementation of the thesis model (`RLPolicyStrategy`).
* Integrates inference logic:

  1. Extracts features (BBO, spread, inventory).
  2. Calls `inference_stub()`.
  3. Generates and queues orders.
* `inference_stub()` – Placeholder for real model (e.g., PyTorch, ONNX, FPGA driver).

---

### `include/ultra/risk/`

**pretrade_checker.hpp / .cpp**

* Performs synchronous order-level risk checks:

  * Max order size
  * Max position
  * Max notional exposure
* Returns `false` for rejected orders.

---

### `include/ultra/execution/`

**gateway_sim.hpp / .cpp**

* Simulated exchange / matching engine.
* `send_order()` sends accept messages.
* `try_match()` performs basic cross-matching for fills.
* Supports feedback loop to strategy via `exec_reports_queue_`.

---

## Future Work

1. **High-Performance Order Book**

   * Replace vector-based book with O(1) cache-optimized arrays.

2. **Real Inference**

   * Load and run models from ONNX or libtorch.

3. **FPGA Control Plane**

   * Implement PCIe communication between CPU (C++) and FPGA.

4. **Live Market Data**

   * Replace simulated feed with DPDK or Solarflare-based multicast listeners.

5. **Live Gateway**

   * Replace GatewaySim with FIX or OUCH protocol exchange connectivity.

---

## Summary Outline

```
Ultra HFT: AI-Integrated Trading System
├── Key Features
├── System Architecture
├── How to Build and Run
│   ├── Build Environment
│   ├── Enter Container
│   ├── Build
│   └── Run
├── Component & File Deep Dive
│   ├── apps/live-engine/
│   ├── include/ultra/core/
│   ├── include/ultra/network/
│   ├── include/ultra/market-data/
│   ├── include/ultra/strategy/
│   ├── include/ultra/risk/
│   └── include/ultra/execution/
└── Future Work
```

```
