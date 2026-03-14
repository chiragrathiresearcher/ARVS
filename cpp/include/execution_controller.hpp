/**
 * execution_controller.hpp
 * ARVS C++ Execution Controller — real-time actuator command loop
 *
 * Responsibilities:
 *   1. Receive an MVISequence from the Python planner (via ROS2 topic)
 *   2. For each action, pass it through SafetyGate before issuing
 *   3. Monitor divergence between predicted and actual RobotState
 *   4. Trigger replanning (via ROS2 service) if divergence exceeds threshold
 *   5. Pet the WatchdogTimer every control cycle
 *   6. Activate emergency stop if a critical violation occurs mid-execution
 *
 * The control loop runs at CONTROL_LOOP_HZ in a dedicated SCHED_FIFO thread
 * (priority 80) to ensure deterministic timing on Linux.
 * On RTOS this maps to the highest-priority task in the kernel.
 */

#pragma once
#include "arvs_types.hpp"
#include "safety_gate.hpp"
#include "watchdog_timer.hpp"
#include <atomic>
#include <cstdint>

namespace arvs {

static constexpr double   CONTROL_LOOP_HZ        = 10.0;
static constexpr double   CONTROL_LOOP_PERIOD_S  = 1.0 / CONTROL_LOOP_HZ;
static constexpr uint32_t MAX_SEQUENCE_ACTIONS   = 64;

// ─────────────────────────────────────────────
//  MVI Sequence (fixed-size, no heap)
// ─────────────────────────────────────────────
struct MVISequence {
    char     sequence_id[64]{};
    Action   actions[MAX_SEQUENCE_ACTIONS]{};
    uint32_t n_actions{0};
    double   expected_duration{0.0};  // s
    double   predicted_risk{0.0};     // [0,1]
};

// ─────────────────────────────────────────────
//  Execution result
// ─────────────────────────────────────────────
enum class ExecStatus : uint8_t {
    SUCCESS         = 0,
    PARTIAL_SUCCESS = 1,
    FAILED          = 2,
    SAFETY_STOPPED  = 3,
    CANCELLED       = 4
};

struct ActionResult {
    char       action_id[ACTION_ID_LEN]{};
    ExecStatus status{ExecStatus::FAILED};
    double     expected_duration{0.0};
    double     actual_duration{0.0};
    double     position_error{0.0};   // m
    double     orientation_error{0.0};
    bool       safety_gate_blocked{false};
};

struct SequenceResult {
    char         sequence_id[64]{};
    ExecStatus   status{ExecStatus::FAILED};
    ActionResult action_results[MAX_SEQUENCE_ACTIONS]{};
    uint32_t     n_results{0};
    double       total_duration{0.0};
};

// ─────────────────────────────────────────────
//  Divergence thresholds
// ─────────────────────────────────────────────
struct DivergenceThresholds {
    double position_m{0.10};         // m
    double orientation_rad{0.05};    // rad (quaternion dot < 1 - thr)
    double velocity_ms{0.20};        // m/s
    double temperature_k{2.0};       // K
    double power_w{10.0};            // W
};

// ─────────────────────────────────────────────
//  Actuator interface — implement for your hardware
// ─────────────────────────────────────────────
class ActuatorInterface {
public:
    virtual ~ActuatorInterface() = default;
    /** Send command to hardware. Returns false if hardware rejects. */
    virtual bool send_command(const Action& action,
                              const RobotState& current_state) = 0;
    /** Read current hardware state into out_state. */
    virtual bool read_state(RobotState& out_state) = 0;
    /** Immediate hardware stop — must be interrupt-safe. */
    virtual void emergency_stop() = 0;
};

// ─────────────────────────────────────────────
//  Execution Controller
// ─────────────────────────────────────────────
class ExecutionController {
public:
    ExecutionController(const char*        robot_id,
                        SafetyGate&        gate,
                        WatchdogTimer&     watchdog,
                        ActuatorInterface* hw_interface = nullptr);

    /**
     * execute_sequence — blocking call.
     * Runs each action through the gate, then sends to hardware.
     * Returns after all actions complete or on first failure.
     */
    SequenceResult execute_sequence(const MVISequence& seq,
                                    const RobotState&  initial_state);

    /** Abort current sequence immediately. Safe to call from any thread. */
    void request_stop();

    /** Check if currently executing. */
    bool is_executing() const { return executing_.load(); }

    /** Performance counters published to /arvs/exec_metrics. */
    struct Metrics {
        uint64_t sequences_run{0};
        uint64_t actions_run{0};
        uint64_t gate_blocks{0};
        uint64_t emergency_stops{0};
        uint64_t replan_requests{0};
        double   avg_position_error_m{0.0};
    };
    Metrics metrics() const { return metrics_; }

private:
    char              robot_id_[ROBOT_ID_LEN]{};
    SafetyGate&       gate_;
    WatchdogTimer&    watchdog_;
    ActuatorInterface* hw_;

    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> executing_{false};
    mutable Metrics   metrics_{};

    DivergenceThresholds divergence_thresholds_{};

    // Execute a single action within a sequence
    ActionResult execute_action(const Action&     action,
                                const RobotState& current_state,
                                const char*       sequence_id);

    // Physics-only state predictor (no random, no heap)
    RobotState predict_state(const RobotState& s, const Action& a,
                             double progress) const;

    // Divergence check between expected and actual state
    bool check_divergence(const RobotState& expected,
                          const RobotState& actual,
                          ActionResult&     result) const;

    // Update running average position error
    void update_metrics(const ActionResult& r);

    // Real-time sleep for remainder of control period
    static void sleep_remainder(double period_s, double elapsed_s);
};

} // namespace arvs
