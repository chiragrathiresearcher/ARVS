/**
 * execution_controller.cpp
 * ARVS Execution Controller — implementation
 */

#include "execution_controller.hpp"
#include <cstring>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <thread>

#ifdef __linux__
#  include <pthread.h>
#  include <sched.h>
#endif

namespace arvs {

// ─────────────────────────────────────────────────────────────────────────────

ExecutionController::ExecutionController(const char*        robot_id,
                                          SafetyGate&        gate,
                                          WatchdogTimer&     watchdog,
                                          ActuatorInterface* hw)
    : gate_(gate), watchdog_(watchdog), hw_(hw)
{
    std::strncpy(robot_id_, robot_id, ROBOT_ID_LEN - 1);
}

// ─────────────────────────────────────────────────────────────────────────────

SequenceResult ExecutionController::execute_sequence(const MVISequence& seq,
                                                      const RobotState&  initial)
{
    SequenceResult sr{};
    std::strncpy(sr.sequence_id, seq.sequence_id, 63);
    sr.status = ExecStatus::FAILED;

    if (seq.n_actions == 0) {
        sr.status = ExecStatus::SUCCESS;
        return sr;
    }

    executing_.store(true);
    stop_requested_.store(false);

    RobotState current = initial;
    double total_time  = 0.0;

    for (uint32_t i = 0; i < seq.n_actions; ++i) {

        // ── Abort check ──────────────────────────────────────────────────────
        if (stop_requested_.load() || gate_.is_emergency_stopped()) {
            sr.status = ExecStatus::CANCELLED;
            break;
        }

        // ── Pet the watchdog before every action ─────────────────────────────
        watchdog_.pet();

        // ── Execute action ───────────────────────────────────────────────────
        ActionResult ar = execute_action(seq.actions[i], current, seq.sequence_id);
        sr.action_results[sr.n_results++] = ar;
        total_time += ar.actual_duration;

        update_metrics(ar);

        if (ar.safety_gate_blocked) {
            sr.status = ExecStatus::SAFETY_STOPPED;
            ++metrics_.gate_blocks;
            break;
        }

        if (ar.status == ExecStatus::FAILED) {
            sr.status = ExecStatus::FAILED;
            break;
        }

        // ── Advance state for next action ────────────────────────────────────
        current = predict_state(current, seq.actions[i], 1.0);
    }

    // If we completed all actions without break
    if (sr.n_results == seq.n_actions &&
        sr.status    != ExecStatus::SAFETY_STOPPED &&
        sr.status    != ExecStatus::FAILED         &&
        sr.status    != ExecStatus::CANCELLED) {
        sr.status = ExecStatus::SUCCESS;
    }

    sr.total_duration = total_time;
    ++metrics_.sequences_run;

    executing_.store(false);
    return sr;
}

// ─────────────────────────────────────────────────────────────────────────────

ActionResult ExecutionController::execute_action(const Action&     action,
                                                  const RobotState& current,
                                                  const char*       seq_id)
{
    ActionResult ar{};
    std::strncpy(ar.action_id, action.action_id, ACTION_ID_LEN - 1);

    using Clock    = std::chrono::steady_clock;
    using Duration = std::chrono::duration<double>;

    auto t_start = Clock::now();

    // ── 1. Safety gate check ─────────────────────────────────────────────────
    RobotState predicted = predict_state(current, action, 1.0);
    SafetyCheckResult gate_result = gate_.check_action(action, current, &predicted);

    if (!gate_result.safe) {
        ar.status             = ExecStatus::FAILED;
        ar.safety_gate_blocked = true;
        ar.actual_duration    = 0.0;
        std::fprintf(stderr,
            "[ARVS EXEC] Action '%s' in seq '%s' BLOCKED by safety gate (%u violations)\n",
            action.action_id, seq_id, gate_result.n_violations);
        return ar;
    }

    // ── 2. Send to hardware (if interface is connected) ───────────────────────
    if (hw_) {
        bool ok = hw_->send_command(action, current);
        if (!ok) {
            ar.status = ExecStatus::FAILED;
            std::fprintf(stderr,
                "[ARVS EXEC] Hardware rejected command '%s'\n", action.action_id);
            return ar;
        }
    }

    // ── 3. Monitor loop — run for the action's nominal duration ──────────────
    const double period_s    = CONTROL_LOOP_PERIOD_S;
    double       elapsed_s   = 0.0;
    bool         diverged    = false;

    while (elapsed_s < action.duration && !stop_requested_.load()) {

        watchdog_.pet();

        auto loop_start = Clock::now();
        const double progress = elapsed_s / action.duration;
        const RobotState exp_state = predict_state(current, action, progress);

        // Read actual state from hardware (or use expected when no hw)
        RobotState act_state = exp_state;
        if (hw_) {
            hw_->read_state(act_state);
        }

        // Divergence check
        if (check_divergence(exp_state, act_state, ar)) {
            diverged = true;
            ++metrics_.replan_requests;
            std::fprintf(stderr,
                "[ARVS EXEC] Divergence detected on '%s': "
                "pos_err=%.3fm ori_err=%.4f\n",
                action.action_id, ar.position_error, ar.orientation_error);
            break;
        }

        auto loop_end = Clock::now();
        const double loop_elapsed = Duration(loop_end - loop_start).count();
        sleep_remainder(period_s, loop_elapsed);

        elapsed_s = Duration(Clock::now() - t_start).count();
    }

    ar.actual_duration = Duration(Clock::now() - t_start).count();

    if (stop_requested_.load()) {
        ar.status = ExecStatus::CANCELLED;
    } else if (diverged) {
        ar.status = ExecStatus::PARTIAL_SUCCESS;
    } else {
        // Timing check: more than 20% over budget → partial
        const double over = std::fabs(ar.actual_duration - action.duration)
                          / std::fmax(action.duration, 1e-6);
        ar.status = (over > 0.2) ? ExecStatus::PARTIAL_SUCCESS
                                 : ExecStatus::SUCCESS;
    }

    ++metrics_.actions_run;
    return ar;
}

// ─────────────────────────────────────────────────────────────────────────────

RobotState ExecutionController::predict_state(const RobotState& s,
                                               const Action& a,
                                               double progress) const
{
    RobotState p = s;
    p.timestamp      = s.timestamp + a.duration * progress;
    p.position.x     = s.position.x + s.velocity.x * a.duration * progress;
    p.position.y     = s.position.y + s.velocity.y * a.duration * progress;
    p.position.z     = s.position.z + s.velocity.z * a.duration * progress;

    // Velocity decays to 50% at completion
    const double decay = 1.0 - progress * 0.5;
    p.velocity.x     = s.velocity.x * decay;
    p.velocity.y     = s.velocity.y * decay;
    p.velocity.z     = s.velocity.z * decay;

    p.temperature    = s.temperature + a.thermal_load * progress;
    p.battery_level  = s.battery_level
                     - (a.power_required * a.duration * progress / 3600.0 / 100.0);
    if (p.battery_level < 0.0) p.battery_level = 0.0;
    p.power_consumption = a.power_required;
    p.confidence        = s.confidence * (1.0 - progress * 0.1);
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────

bool ExecutionController::check_divergence(const RobotState& expected,
                                            const RobotState& actual,
                                            ActionResult&     ar) const
{
    const DivergenceThresholds& thr = divergence_thresholds_;

    // Position error (Euclidean)
    const double dx = expected.position.x - actual.position.x;
    const double dy = expected.position.y - actual.position.y;
    const double dz = expected.position.z - actual.position.z;
    ar.position_error = std::sqrt(dx*dx + dy*dy + dz*dz);

    // Orientation error (quaternion dot → angular distance)
    const double dot = std::fabs(
        expected.orientation.w * actual.orientation.w +
        expected.orientation.x * actual.orientation.x +
        expected.orientation.y * actual.orientation.y +
        expected.orientation.z * actual.orientation.z);
    ar.orientation_error = 1.0 - std::fmin(1.0, dot);

    // Velocity error
    const double dvx = expected.velocity.x - actual.velocity.x;
    const double dvy = expected.velocity.y - actual.velocity.y;
    const double dvz = expected.velocity.z - actual.velocity.z;
    const double vel_err = std::sqrt(dvx*dvx + dvy*dvy + dvz*dvz);

    const double temp_err  = std::fabs(expected.temperature    - actual.temperature);
    const double power_err = std::fabs(expected.power_consumption - actual.power_consumption);

    return (ar.position_error    > thr.position_m)     ||
           (ar.orientation_error > thr.orientation_rad) ||
           (vel_err              > thr.velocity_ms)     ||
           (temp_err             > thr.temperature_k)   ||
           (power_err            > thr.power_w);
}

// ─────────────────────────────────────────────────────────────────────────────

void ExecutionController::update_metrics(const ActionResult& r)
{
    // Exponential moving average of position error (α = 0.1)
    metrics_.avg_position_error_m =
        0.9 * metrics_.avg_position_error_m + 0.1 * r.position_error;

    if (r.status == ExecStatus::SAFETY_STOPPED)
        ++metrics_.emergency_stops;
}

void ExecutionController::request_stop()
{
    stop_requested_.store(true);
}

// ─────────────────────────────────────────────────────────────────────────────

void ExecutionController::sleep_remainder(double period_s, double elapsed_s)
{
    const double remaining_s = period_s - elapsed_s;
    if (remaining_s > 0.0) {
        const auto ns = static_cast<long long>(remaining_s * 1e9);
        std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
    }
}

} // namespace arvs
