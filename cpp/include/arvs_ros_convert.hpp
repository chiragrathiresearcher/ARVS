/**
 * arvs_ros_convert.hpp
 * Conversion utilities between ARVS C++ types and ROS2 message types
 *
 * Every conversion is a pure function — no side effects, no allocations
 * beyond what the ROS2 message type requires.
 */

#pragma once
#include "arvs_types.hpp"

// Forward-declare ROS2 message types
// (The actual headers come from the arvs_msgs package)
#include <arvs_msgs/msg/robot_state_msg.hpp>
#include <arvs_msgs/msg/action_msg.hpp>
#include <arvs_msgs/msg/mvi_sequence_msg.hpp>
#include <arvs_msgs/msg/safety_check_result_msg.hpp>
#include <arvs_msgs/msg/axiom_system_state_msg.hpp>
#include <arvs_msgs/msg/axiom_validation_result_msg.hpp>
#include <arvs_msgs/msg/sequence_result_msg.hpp>

#include <cstring>

namespace arvs {

// ── RobotState ───────────────────────────────────────────────────────────────

inline RobotState ros_to_robot_state(
    const arvs_msgs::msg::RobotStateMsg& m)
{
    RobotState s{};
    std::strncpy(s.robot_id, m.robot_id.c_str(), ROBOT_ID_LEN - 1);
    s.timestamp          = m.timestamp;
    s.position           = {m.position[0], m.position[1], m.position[2]};
    s.velocity           = {m.velocity[0], m.velocity[1], m.velocity[2]};
    s.orientation        = {m.orientation[0], m.orientation[1],
                            m.orientation[2], m.orientation[3]};
    s.angular_velocity   = {m.angular_velocity[0], m.angular_velocity[1],
                            m.angular_velocity[2]};
    s.temperature        = m.temperature;
    s.battery_level      = m.battery_level;
    s.power_consumption  = m.power_consumption;
    s.confidence         = m.confidence;

    s.n_joints = static_cast<uint32_t>(
        std::min(m.joint_names.size(),
                 static_cast<size_t>(MAX_JOINTS)));
    for (uint32_t i = 0; i < s.n_joints; ++i) {
        std::strncpy(s.joints[i].name, m.joint_names[i].c_str(), 31);
        s.joints[i].position = m.joint_positions[i];
        s.joints[i].velocity = m.joint_velocities[i];
        s.joints[i].torque   = m.joint_torques[i];
    }
    return s;
}

// ── Action ───────────────────────────────────────────────────────────────────

inline Action ros_to_action(const arvs_msgs::msg::ActionMsg& m)
{
    Action a{};
    std::strncpy(a.action_id,   m.action_id.c_str(),   ACTION_ID_LEN - 1);
    std::strncpy(a.action_type, m.action_type.c_str(), 31);
    a.duration       = m.duration;
    a.max_torque     = m.max_torque;
    a.max_velocity   = m.max_velocity;
    a.thermal_load   = m.thermal_load;
    a.power_required = m.power_required;
    a.is_reversible  = m.is_reversible;
    a.priority       = m.priority;
    return a;
}

// ── MVISequence ──────────────────────────────────────────────────────────────

inline MVISequence ros_to_mvi_sequence(
    const arvs_msgs::msg::MVISequenceMsg& m)
{
    MVISequence seq{};
    std::strncpy(seq.sequence_id, m.sequence_id.c_str(), 63);
    seq.expected_duration = m.expected_duration;
    seq.predicted_risk    = m.predicted_risk;
    seq.n_actions = static_cast<uint32_t>(
        std::min(m.actions.size(),
                 static_cast<size_t>(MAX_SEQUENCE_ACTIONS)));
    for (uint32_t i = 0; i < seq.n_actions; ++i)
        seq.actions[i] = ros_to_action(m.actions[i]);
    return seq;
}

// ── SafetyCheckResult ────────────────────────────────────────────────────────

inline arvs_msgs::msg::SafetyCheckResultMsg safety_result_to_ros(
    const SafetyCheckResult& r)
{
    arvs_msgs::msg::SafetyCheckResultMsg msg;
    msg.safe       = r.safe;
    msg.confidence = r.confidence;
    msg.n_violations = r.n_violations;
    for (uint32_t i = 0; i < r.n_violations; ++i) {
        msg.violation_types.push_back(
            static_cast<uint8_t>(r.violations[i].type));
        msg.violation_components.push_back(r.violations[i].component);
        msg.violation_actuals.push_back(r.violations[i].actual);
        msg.violation_limits.push_back(r.violations[i].limit);
    }
    return msg;
}

// ── AxiomSystemState ─────────────────────────────────────────────────────────

inline AxiomSystemState ros_to_axiom_state(
    const arvs_msgs::msg::AxiomSystemStateMsg& m)
{
    AxiomSystemState s{};
    s.confidence               = m.confidence;
    s.uncertainty_explicit     = m.uncertainty_explicit;
    s.uncertainty_current      = m.uncertainty_current;
    s.uncertainty_max_credible = m.uncertainty_max_credible;
    s.uncertainty_previous     = m.uncertainty_previous;
    s.has_new_evidence         = m.has_new_evidence;
    s.belief_oscillation_rate  = m.belief_oscillation_rate;
    s.belief_timestamp         = m.belief_timestamp;
    s.belief_validity_window   = m.belief_validity_window;
    s.n_active_authorities     = m.n_active_authorities;
    s.authority_revoked_this_cycle = m.authority_revoked_this_cycle;
    s.authority_explicitly_defined = m.authority_explicitly_defined;
    s.is_acting                = m.is_acting;
    std::strncpy(s.evaluation_basis, m.evaluation_basis.c_str(), 31);
    s.potential_harm           = m.potential_harm;
    s.action_is_reversible     = m.action_is_reversible;
    s.distance_to_harm         = m.distance_to_harm;
    s.constraint_tightness     = m.constraint_tightness;
    s.refusal_is_illegal       = m.refusal_is_illegal;
    s.action_is_full_capability = m.action_is_full_capability;
    s.is_safe_for_full_capability = m.is_safe_for_full_capability;
    s.is_optimizing            = m.is_optimizing;
    s.is_safe                  = m.is_safe;
    s.all_actions_gated        = m.all_actions_gated;
    s.gate_decision_final      = m.gate_decision_final;
    s.learning_overrides       = m.learning_overrides;
    s.is_irreversible_context  = m.is_irreversible_context;
    s.online_learning_active   = m.online_learning_active;
    s.action_has_explanation   = m.action_has_explanation;
    s.justification_timestamp  = m.justification_timestamp;
    s.action_timestamp         = m.action_timestamp;
    s.system_mode              = static_cast<SystemMode>(m.system_mode);
    return s;
}

// ── AxiomValidationResult ────────────────────────────────────────────────────

inline arvs_msgs::msg::AxiomValidationResultMsg axiom_result_to_ros(
    const AxiomValidationResult& r)
{
    arvs_msgs::msg::AxiomValidationResultMsg msg;
    msg.timestamp        = r.timestamp;
    msg.authority_valid  = r.authority_valid;
    msg.action_permitted = r.action_permitted;
    msg.n_checked        = r.n_checked;
    for (uint32_t i = 0; i < r.n_checked; ++i) {
        msg.axiom_ids.push_back(r.checks[i].axiom_id);
        msg.axiom_passed.push_back(r.checks[i].passed);
        msg.axiom_reasons.push_back(r.checks[i].reason);
    }
    return msg;
}

// ── SequenceResult ───────────────────────────────────────────────────────────

inline arvs_msgs::msg::SequenceResultMsg sequence_result_to_ros(
    const SequenceResult& r)
{
    arvs_msgs::msg::SequenceResultMsg msg;
    msg.sequence_id    = r.sequence_id;
    msg.status         = static_cast<uint8_t>(r.status);
    msg.total_duration = r.total_duration;
    msg.n_results      = r.n_results;
    for (uint32_t i = 0; i < r.n_results; ++i) {
        msg.action_ids.push_back(r.action_results[i].action_id);
        msg.action_statuses.push_back(
            static_cast<uint8_t>(r.action_results[i].status));
        msg.position_errors.push_back(r.action_results[i].position_error);
        msg.gate_blocked.push_back(r.action_results[i].safety_gate_blocked);
    }
    return msg;
}

} // namespace arvs
