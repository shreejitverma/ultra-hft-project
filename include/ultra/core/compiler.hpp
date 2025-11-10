#pragma once

// Branch prediction hints
#define ULTRA_LIKELY(x)   __builtin_expect(!!(x), 1)
#define ULTRA_UNLIKELY(x) __builtin_expect(!!(x), 0)

// Force inlining
#define ULTRA_ALWAYS_INLINE __attribute__((always_inline)) inline
#define ULTRA_NEVER_INLINE  __attribute__((noinline))

// Hot/cold path hints
#define ULTRA_HOT   __attribute__((hot))
#define ULTRA_COLD  __attribute__((cold))

// Prefetch hints
#define ULTRA_PREFETCH_READ(addr)  __builtin_prefetch(addr, 0, 3)
#define ULTRA_PREFETCH_WRITE(addr) __builtin_prefetch(addr, 1, 3)

// Cache line size
#define ULTRA_CACHE_LINE_SIZE 64
#define ULTRA_CACHE_ALIGNED alignas(ULTRA_CACHE_LINE_SIZE)

// Memory barriers
#define ULTRA_COMPILER_BARRIER() asm volatile("" ::: "memory")
#define ULTRA_MEMORY_BARRIER()   __sync_synchronize()

// Restrict pointer
#define ULTRA_RESTRICT __restrict__

// Assume hint for optimizer
#define ULTRA_ASSUME(x) do { if (!(x)) __builtin_unreachable(); } while(0)
