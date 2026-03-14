/**
 * safety_gate.cpp
 * ARVS Safety Gate — implementation
 */

#include "safety_gate.hpp"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <chrono>

namespace arvs {

// ─────────────────────────────────────────────────────────────────────────────
SafetyGate::SafetyGate(const SafetyConstraints& constraints)
    : constraints_(constraints)
{}

// ─────────────────────────────────────────────────────────────────────────────
SafetyCheckResult SafetyGate::check_action(const Action&     action,
                                            const RobotState& current,
                                            const RobotState* predicted) const
{
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    SafetyCheckResult result{};
    result.confidence = current.confidence;
    if (predicted == nullptr) result.confidence *= 0.7;

    // ── Hard latch: emergency stop overrides everything ──────────────────────
    if (emergency_latched_) {
        result.safe = false;
        result.add_violation(ViolationType::AXIOM_VIOLATION,
                             "emergency_latch", 1.0, 0.0);
        ++metrics_.total_checks;
        ++metrics_.total_blocks;
        std::snprintf(metrics_.last_block_reason,
                      sizeof(metrics_.last_block_reason),
                      "emergency latch: %s", emergency_reason_);
        return result;
    }

    // ── Run all constraint checkers ──────────────────────────────────────────
    check_torque        (action, current, result);
    check_thermal       (action, current, result);
    check_structural    (action, current, result);
    check_power         (action, current, result);
    check_joint_limits  (current, result);

    const RobotState* check_state = (predicted != nullptr) ? predicted : &current;
    check_hazard_proximity(&current, check_state, result);
    check_comm_blackout (action, current, result);

    result.safe = (result.n_violations == 0);

    // ── Snapshot last known safe state ──────────────────────────────────────
    if (result.safe) {
        last_safe_state_ = current;
    }

    // ── Metrics ─────────────────────────────────────────────────────────────
    auto t1 = Clock::now();
    metrics_.last_check_duration_us =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    ++metrics_.total_checks;
    if (!result.safe) {
        ++metrics_.total_blocks;
        if (result.n_violations > 0) {
            std::snprintf(metrics_.last_block_reason,
                          sizeof(metrics_.last_block_reason),
                          "violation[0]: type=%u component=%s actual=%.3f limit=%.3f",
                          static_cast<unsigned>(result.violations[0].type),
                          result.violations[0].component,
                          result.violations[0].actual,
                          result.violations[0].limit);
        }
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
void SafetyGate::emergency_stop(const char* reason)
{
    emergency_latched_ = true;
    std::strncpy(emergency_reason_, reason, sizeof(emergency_reason_) - 1);
    // Write to stderr — on a real flight computer this goes to the
    // mission event log via the audit logger
    std::fprintf(stderr, "[ARVS SAFETY GATE] EMERGENCY STOP: %s\n", reason);
}

void SafetyGate::reset_emergency_stop()
{
    emergency_latched_ = false;
    std::memset(emergency_reason_, 0, sizeof(emergency_reason_));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Constraint checkers
// ─────────────────────────────────────────────────────────────────────────────

void SafetyGate::check_torque(const Action& action,
                               const RobotState& /*state*/,
                               SafetyCheckResult& out) const
{
    if (action.max_torque < 0.0) return;  // not specified → skip

    for (uint32_t i = 0; i < constraints_.n_joints; ++i) {
        const double safe_limit =
            constraints_.joints[i].max_torque * SAFETY_MARGIN_TORQUE;
        if (action.max_torque > safe_limit) {
            out.add_violation(ViolationType::TORQUE_EXCEEDED,
                              constraints_.joints[i].name,
                              action.max_torque, safe_limit);
        }
    }
}

void SafetyGate::check_thermal(const Action& action,
                                const RobotState& state,
                                SafetyCheckResult& out) const
{
    const double predicted_temp = state.temperature + action.thermal_load;

    for (uint32_t i = 0; i < constraints_.n_thermals; ++i) {
        const double safe_limit =
            constraints_.thermals[i].max_temperature * SAFETY_MARGIN_THERMAL;
        if (predicted_temp > safe_limit) {
            out.add_violation(ViolationType::TEMPERATURE_EXCEEDED,
                              constraints_.thermals[i].component,
                              predicted_temp, safe_limit);
        }
    }
}

void SafetyGate::check_structural(const Action& action,
                                   const RobotState& /*state*/,
                                   SafetyCheckResult& out) const
{
    // Composite load factor derived from torque + velocity + duration
    double load = 0.0;
    int    n    = 0;

    if (action.max_torque > 0.0)   { load += action.max_torque  / MAX_TORQUE_NM;  ++n; }
    if (action.max_velocity > 0.0) { load += action.max_velocity / MAX_VELOCITY_MS; ++n; }
    if (action.duration > 0.0)     { load += std::fmin(1.0, action.duration / 30.0) * 0.5; ++n; }

    if (n == 0) return;
    load /= static_cast<double>(n);

    const double max_load = 1.0 / SAFETY_MARGIN_STRUCT;
    if (load > max_load) {
        out.add_violation(ViolationType::LOAD_FACTOR_EXCEEDED,
                          "structure", load, max_load);
    }
}

void SafetyGate::check_power(const Action& action,
                              const RobotState& state,
                              SafetyCheckResult& out) const
{
    if (action.power_required <= 0.0) return;

    // Energy in Wh drawn by this action
    const double energy_wh  = action.power_required * action.duration / 3600.0;
    const double capacity   = 100.0;  // normalised to 100 Wh equivalent
    const double post_batt  = state.battery_level - (energy_wh / capacity);
    const double min_batt   = constraints_.min_battery + SAFETY_MARGIN_POWER;

    if (post_batt < min_batt) {
        out.add_violation(ViolationType::BATTERY_LOW,
                          "battery", post_batt, min_batt);
    }
}

void SafetyGate::check_joint_limits(const RobotState& state,
                                     SafetyCheckResult& out) const
{
    for (uint32_t si = 0; si < state.n_joints; ++si) {
        const JointState& js = state.joints[si];

        // Find matching limit entry
        for (uint32_t li = 0; li < constraints_.n_joints; ++li) {
            if (std::strncmp(js.name, constraints_.joints[li].name, 31) != 0)
                continue;

            const JointLimits& lim = constraints_.joints[li];
            const double margin    = 0.1;  // rad soft margin

            if (js.position < lim.min_position + margin) {
                out.add_violation(ViolationType::JOINT_LIMIT,
                                  js.name, js.position,
                                  lim.min_position + margin);
            } else if (js.position > lim.max_position - margin) {
                out.add_violation(ViolationType::JOINT_LIMIT,
                                  js.name, js.position,
                                  lim.max_position - margin);
            }
            break;
        }
    }
}

void SafetyGate::check_hazard_proximity(const RobotState* current,
                                         const RobotState* predicted,
                                         SafetyCheckResult& out) const
{
    const Vec3& pos = (predicted != nullptr) ? predicted->position
                                             : current->position;

    for (uint32_t i = 0; i < constraints_.n_zones; ++i) {
        const CollisionZone& zone = constraints_.zones[i];

        // Hard exclusion: inside the cylinder
        if (point_in_cylinder(pos, zone)) {
            out.add_violation(ViolationType::HAZARD_PROXIMITY,
                              zone.name, 0.0, zone.radius);
            continue;
        }

        // Soft exclusion: inside the safe-distance shell
        if (zone.safe_distance > 0.0) {
            const double dx  = pos.x - zone.centre.x;
            const double dy  = pos.y - zone.centre.y;
            const double d2d = std::sqrt(dx*dx + dy*dy);
            const double clearance = d2d - zone.radius;
            if (clearance < zone.safe_distance) {
                out.add_violation(ViolationType::HAZARD_PROXIMITY,
                                  zone.name, clearance, zone.safe_distance);
            }
        }
    }
}

void SafetyGate::check_comm_blackout(const Action& action,
                                      const RobotState& state,
                                      SafetyCheckResult& out) const
{
    const double action_end = state.timestamp + action.duration;

    for (uint32_t i = 0; i < constraints_.n_blackouts; ++i) {
        const double bs = constraints_.comm_blackout_start[i];
        const double be = constraints_.comm_blackout_end[i];

        if (state.timestamp < be && action_end > bs) {
            // Allow safety-type or low-power short actions
            bool allowed = (std::strncmp(action.action_type, "safety", 6) == 0);
            if (!allowed)
                allowed = (action.power_required < 20.0 && action.duration < 5.0);

            if (!allowed) {
                out.add_violation(ViolationType::COMM_BLACKOUT,
                                  "communication",
                                  action.duration, 0.0);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Geometry helpers
// ─────────────────────────────────────────────────────────────────────────────

double SafetyGate::dist_sq_2d(const Vec3& a, const Vec3& b) const
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return dx*dx + dy*dy;
}

bool SafetyGate::point_in_cylinder(const Vec3& pt,
                                    const CollisionZone& z) const
{
    const double dx  = pt.x - z.centre.x;
    const double dy  = pt.y - z.centre.y;
    const double d2d = std::sqrt(dx*dx + dy*dy);
    const double dz  = std::fabs(pt.z - z.centre.z);
    return (d2d <= z.radius) && (dz <= z.height / 2.0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Physics predictor
// ─────────────────────────────────────────────────────────────────────────────

RobotState SafetyGate::predict_state(const RobotState& s,
                                      const Action& a) const
{
    RobotState p = s;
    p.timestamp      = s.timestamp + a.duration;
    p.position.x     = s.position.x + s.velocity.x * a.duration;
    p.position.y     = s.position.y + s.velocity.y * a.duration;
    p.position.z     = s.position.z + s.velocity.z * a.duration;
    p.temperature    = s.temperature + a.thermal_load;
    p.battery_level  = s.battery_level
                     - (a.power_required * a.duration / 3600.0 / 100.0);
    if (p.battery_level < 0.0) p.battery_level = 0.0;
    p.power_consumption = a.power_required;
    p.confidence        = s.confidence * 0.9;
    return p;
}

} // namespace arvs
