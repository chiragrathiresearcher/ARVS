/**
 * watchdog_timer.hpp
 * ARVS C++ Watchdog Timer — hardware-reset trigger
 *
 * Implements a "pet-the-dog" pattern:
 *   - Main control loop must call pet() every TIMEOUT_MS milliseconds
 *   - If pet() is not called in time, on_timeout() fires
 *   - on_timeout() triggers: emergency_stop → safe_hold → hardware reset
 *
 * On Linux flight computers: sends SIGTERM to the ARVS process and
 * writes a tombstone to /var/log/arvs/watchdog_reset.log.
 *
 * On RTOS / bare-metal: replace on_timeout() with a direct hardware
 * watchdog register write (target-specific BSP layer).
 *
 * Thread safety: pet() is async-signal-safe, callable from any thread.
 */

#pragma once
#include "arvs_types.hpp"
#include <atomic>
#include <chrono>
#include <thread>
#include <functional>
#include <cstdint>

namespace arvs {

class WatchdogTimer {
public:
    using TimeoutCallback = std::function<void(const char* reason)>;

    /**
     * @param timeout_ms   Maximum interval between pet() calls (ms)
     * @param on_timeout   Called from watchdog thread when deadline missed
     */
    explicit WatchdogTimer(uint32_t timeout_ms = 500,
                           TimeoutCallback on_timeout = nullptr);
    ~WatchdogTimer();

    // Non-copyable
    WatchdogTimer(const WatchdogTimer&) = delete;
    WatchdogTimer& operator=(const WatchdogTimer&) = delete;

    /** Start the watchdog monitoring thread. */
    void start();

    /** Stop the watchdog (call during clean shutdown only). */
    void stop();

    /**
     * pet — reset the deadline.
     * Must be called from the main control loop at every iteration.
     * Async-signal-safe: uses std::atomic store.
     */
    void pet();

    /**
     * force_reset — trigger the timeout callback immediately.
     * Called by SafetyGate::emergency_stop() or AxiomValidator
     * when a CRITICAL axiom violation is detected.
     */
    void force_reset(const char* reason);

    /** Is the watchdog healthy (has been pet recently)? */
    bool is_healthy() const;

    uint64_t missed_deadlines() const { return missed_deadlines_.load(); }
    uint64_t total_pets()       const { return total_pets_.load(); }

private:
    uint32_t         timeout_ms_;
    TimeoutCallback  on_timeout_cb_;

    std::atomic<bool>     running_{false};
    std::atomic<uint64_t> last_pet_ns_{0};   // nanoseconds since epoch
    std::atomic<uint64_t> missed_deadlines_{0};
    std::atomic<uint64_t> total_pets_{0};

    std::thread monitor_thread_;

    void monitor_loop();

    static uint64_t now_ns();
    static void write_tombstone(const char* reason);

    static constexpr const char* TOMBSTONE_PATH =
        "/var/log/arvs/watchdog_reset.log";
};

} // namespace arvs
