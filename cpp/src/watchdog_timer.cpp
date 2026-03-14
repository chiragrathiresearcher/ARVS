/**
 * watchdog_timer.cpp
 * ARVS Watchdog Timer — implementation
 */

#include "watchdog_timer.hpp"
#include <cstdio>
#include <cstring>
#include <ctime>
#include <chrono>
#include <thread>

// POSIX headers for tombstone write and process signalling
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <csignal>

namespace arvs {

// ─────────────────────────────────────────────────────────────────────────────

WatchdogTimer::WatchdogTimer(uint32_t timeout_ms, TimeoutCallback on_timeout)
    : timeout_ms_(timeout_ms)
    , on_timeout_cb_(std::move(on_timeout))
{}

WatchdogTimer::~WatchdogTimer()
{
    stop();
}

void WatchdogTimer::start()
{
    if (running_.exchange(true)) return;  // already running
    last_pet_ns_.store(now_ns());
    monitor_thread_ = std::thread(&WatchdogTimer::monitor_loop, this);

// Set SCHED_FIFO priority on Linux for hard real-time behaviour
#ifdef __linux__
    struct sched_param param{};
    param.sched_priority = 90;  // highest below kernel threads
    pthread_setschedparam(monitor_thread_.native_handle(),
                          SCHED_FIFO, &param);
#endif
}

void WatchdogTimer::stop()
{
    if (!running_.exchange(false)) return;
    if (monitor_thread_.joinable()) monitor_thread_.join();
}

void WatchdogTimer::pet()
{
    last_pet_ns_.store(now_ns());
    ++total_pets_;
}

bool WatchdogTimer::is_healthy() const
{
    const uint64_t elapsed_ms =
        (now_ns() - last_pet_ns_.load()) / 1'000'000ULL;
    return elapsed_ms < static_cast<uint64_t>(timeout_ms_);
}

void WatchdogTimer::force_reset(const char* reason)
{
    write_tombstone(reason);
    if (on_timeout_cb_) {
        on_timeout_cb_(reason);
    } else {
        // Default: SIGTERM the current process so the OS can restart it
        std::fprintf(stderr,
            "[ARVS WATCHDOG] force_reset: %s — sending SIGTERM\n", reason);
        std::raise(SIGTERM);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Monitor loop — runs in dedicated thread at SCHED_FIFO priority
// ─────────────────────────────────────────────────────────────────────────────

void WatchdogTimer::monitor_loop()
{
    // Check interval = timeout / 4 to catch deadline miss promptly
    const uint64_t timeout_ns  = static_cast<uint64_t>(timeout_ms_) * 1'000'000ULL;
    const uint64_t check_ns    = timeout_ns / 4;
    const auto     sleep_dur   = std::chrono::nanoseconds(check_ns);

    while (running_.load()) {
        std::this_thread::sleep_for(sleep_dur);

        if (!running_.load()) break;

        const uint64_t elapsed = now_ns() - last_pet_ns_.load();
        if (elapsed > timeout_ns) {
            ++missed_deadlines_;

            char reason[256];
            std::snprintf(reason, sizeof(reason),
                "watchdog timeout: elapsed=%.0f ms, limit=%u ms",
                static_cast<double>(elapsed) / 1e6,
                timeout_ms_);

            write_tombstone(reason);

            if (on_timeout_cb_) {
                on_timeout_cb_(reason);
            } else {
                std::fprintf(stderr,
                    "[ARVS WATCHDOG] TIMEOUT: %s — sending SIGTERM\n", reason);
                std::raise(SIGTERM);
            }

            // Reset the pet timer so we don't fire every check after timeout
            last_pet_ns_.store(now_ns());
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

uint64_t WatchdogTimer::now_ns()
{
    using Clock = std::chrono::steady_clock;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now().time_since_epoch()).count());
}

void WatchdogTimer::write_tombstone(const char* reason)
{
    // Create log directory if needed
    ::mkdir("/var/log/arvs", 0755);

    int fd = ::open(TOMBSTONE_PATH,
                    O_WRONLY | O_CREAT | O_APPEND,
                    0644);
    if (fd < 0) {
        // Fallback: stderr only
        std::fprintf(stderr,
            "[ARVS WATCHDOG] tombstone write failed, reason: %s\n", reason);
        return;
    }

    // Get wall-clock timestamp
    std::time_t now_t = std::time(nullptr);
    char ts[64]{};
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now_t));

    char line[512];
    int  len = std::snprintf(line, sizeof(line),
        "[%s] ARVS WATCHDOG RESET: %s\n", ts, reason);

    // write() is async-signal-safe
    ::write(fd, line, static_cast<size_t>(len > 0 ? len : 0));
    ::close(fd);
}

} // namespace arvs
