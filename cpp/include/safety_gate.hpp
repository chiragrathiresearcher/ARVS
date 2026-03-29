/**
 * safety_gate.hpp
 * ARVS C++ Safety Gate — deterministic hard-stop, sub-millisecond path
 *
 * The Safety Gate is the LAST line of defence before any actuator command
 * leaves the flight computer.  It must:
 *   - Run in bounded time (no dynamic allocation, no recursion)
 *   - Be callable from a real-time thread at 100 Hz or above
 *   - Never return "safe" when it should block
 *   - Mirror the Python SafetyGate logic exactly (same margins, same checks)
 *
 * ROS2 integration: the gate is wrapped in safety_gate_node.cpp which
 * subscribes to /arvs/proposed_action and publishes to /arvs/gate_result.
 */

#pragma once
#include "arvs_types.hpp"
#include <cstdint>

namespace arvs {

// Per-joint torque limit table (populated from ROS2 param server at init)
struct JointLimits {
    char   name[32]{};
    double max_torque{MAX_TORQUE_NM};     // N·m
    double max_velocity{MAX_VELOCITY_MS}; // rad/s
    double min_position{-3.14159};        // rad
    double max_position{ 3.14159};        // rad
};

struct ThermalLimit {
    char   component[32]{};
    double max_temperature{MAX_TEMPERATURE_K};  // K
};

struct CollisionZone {
    char   name[32]{};
    Vec3   centre{};
    double radius{0.0};        // m
    double height{10.0};       // m
    double safe_distance{0.0}; // m — warn before entering
};

static constexpr uint32_t MAX_JOINT_LIMITS    = 16;
static constexpr uint32_t MAX_THERMAL_LIMITS  = 8;
static constexpr uint32_t MAX_COLLISION_ZONES = 16;
static constexpr uint32_t MAX_COMM_BLACKOUTS  = 8;

struct SafetyConstraints {
    JointLimits    joints[MAX_JOINT_LIMITS]{};
    uint32_t       n_joints{0};

    ThermalLimit   thermals[MAX_THERMAL_LIMITS]{};
    uint32_t       n_thermals{0};

    CollisionZone  zones[MAX_COLLISION_ZONES]{};
    uint32_t       n_zones{0};

    double         comm_blackout_start[MAX_COMM_BLACKOUTS]{};
    double         comm_blackout_end[MAX_COMM_BLACKOUTS]{};
    uint32_t       n_blackouts{0};

    double         min_battery{MIN_BATTERY_FRACTION};
};

// ─────────────────────────────────────────────
class SafetyGate {
public:
    explicit SafetyGate(const SafetyConstraints& constraints);

    /**
     * check_action — the primary hot-path call.
     * Called once per proposed action before it reaches the actuator.
     * Must complete in < 500 µs on a Cortex-A53 at 1.2 GHz.
     */
    SafetyCheckResult check_action(const Action&     action,
                                   const RobotState& current,
                                   const RobotState* predicted = nullptr) const;

    /**
     * emergency_stop — called by WatchdogTimer or AxiomValidator.
     * Sets internal latch; all subsequent check_action calls return unsafe
     * until reset_emergency_stop() is called with sufficient authority.
     */
    void emergency_stop(const char* reason);
    bool is_emergency_stopped() const { return emergency_latched_; }
    void reset_emergency_stop();

    /** Returns the last safe state snapshot (copied on each passing check). */
    const RobotState& last_safe_state() const { return last_safe_state_; }

    /** Expose metrics for the /arvs/safety_metrics ROS2 topic. */
    struct Metrics {
        uint64_t total_checks{0};
        uint64_t total_blocks{0};
        double   last_check_duration_us{0.0};
        char     last_block_reason[128]{};
    };
    Metrics metrics() const { return metrics_; }

private:
    SafetyConstraints constraints_;
    mutable RobotState last_safe_state_{};       // updated by const check_action
    mutable bool       emergency_latched_{false}; // latched by emergency_stop
    mutable char       emergency_reason_[128]{};  // set by emergency_stop
    mutable Metrics    metrics_{};

    // Individual constraint checkers — all const, no side-effects
    void check_torque    (const Action&, const RobotState&, SafetyCheckResult&) const;
    void check_thermal   (const Action&, const RobotState&, SafetyCheckResult&) const;
    void check_structural(const Action&, const RobotState&, SafetyCheckResult&) const;
    void check_power     (const Action&, const RobotState&, SafetyCheckResult&) const;
    void check_joint_limits(const RobotState&,              SafetyCheckResult&) const;
    void check_hazard_proximity(const RobotState*, const RobotState*, SafetyCheckResult&) const;
    void check_comm_blackout(const Action&, const RobotState&, SafetyCheckResult&) const;

    // Geometry helpers (no sqrt in the collision check hot path)
    double dist_sq_2d(const Vec3& a, const Vec3& b) const;
    bool   point_in_cylinder(const Vec3& pt, const CollisionZone& z) const;

    // State predictor (purely physics, no random)
    RobotState predict_state(const RobotState& s, const Action& a) const;
};

} // namespace arvs
