/**
 * arvs_types.hpp
 * ARVS Shared Data Types — C++ side
 *
 * Design constraints (space / safety-critical):
 *   - No heap allocation in hot paths (no std::vector, no std::string in structs)
 *   - Fixed-size arrays throughout
 *   - Plain C-compatible layout for future MCU / RTOS ports
 *   - All physical limits defined here as constexpr — single source of truth
 */

#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>

namespace arvs {

// ─────────────────────────────────────────────
//  Compile-time physical limits
// ─────────────────────────────────────────────
static constexpr double MAX_TORQUE_NM          = 100.0;
static constexpr double MAX_VELOCITY_MS        = 2.0;
static constexpr double MAX_TEMPERATURE_K      = 373.0;   // 100 °C
static constexpr double MIN_TEMPERATURE_K      = 123.0;   // -150 °C
static constexpr double MIN_BATTERY_FRACTION   = 0.15;    // 15 %
static constexpr double SAFETY_MARGIN_TORQUE   = 0.80;
static constexpr double SAFETY_MARGIN_THERMAL  = 0.90;
static constexpr double SAFETY_MARGIN_POWER    = 0.20;
static constexpr double SAFETY_MARGIN_STRUCT   = 1.50;
static constexpr double IRREVERSIBLE_CONF_THR  = 0.95;
static constexpr double MAX_OSCILLATION_RATE   = 2.0;     // flips/s

static constexpr uint32_t MAX_JOINTS           = 12;
static constexpr uint32_t MAX_VIOLATIONS       = 32;
static constexpr uint32_t ROBOT_ID_LEN         = 32;
static constexpr uint32_t ACTION_ID_LEN        = 64;

// ─────────────────────────────────────────────
//  System mode
// ─────────────────────────────────────────────
enum class SystemMode : uint8_t {
    NORMAL    = 0,
    DEGRADED  = 1,
    SAFE_HOLD = 2,
    EMERGENCY = 3
};

// ─────────────────────────────────────────────
//  Robot state (POD — no virtuals, no heap)
// ─────────────────────────────────────────────
struct Vec3 {
    double x{0.0}, y{0.0}, z{0.0};
    double norm() const { return std::sqrt(x*x + y*y + z*z); }
};

struct Quat {
    double w{1.0}, x{0.0}, y{0.0}, z{0.0};
    double dot(const Quat& o) const { return w*o.w + x*o.x + y*o.y + z*o.z; }
};

struct JointState {
    char   name[32]{};
    double position{0.0};   // rad
    double velocity{0.0};   // rad/s
    double torque{0.0};     // N·m
};

struct RobotState {
    char     robot_id[ROBOT_ID_LEN]{};
    double   timestamp{0.0};            // UNIX seconds (mission epoch)

    Vec3     position{};                // m, mission frame
    Vec3     velocity{};                // m/s
    Quat     orientation{};
    Vec3     angular_velocity{};        // rad/s

    double   temperature{293.0};        // K
    double   battery_level{1.0};        // [0,1] state-of-charge
    double   power_consumption{0.0};    // W
    double   confidence{1.0};           // [0,1]

    JointState joints[MAX_JOINTS]{};
    uint32_t   n_joints{0};
};

// ─────────────────────────────────────────────
//  Action descriptor
// ─────────────────────────────────────────────
struct Action {
    char     action_id[ACTION_ID_LEN]{};
    char     action_type[32]{};
    double   duration{0.0};             // s
    double   max_torque{-1.0};          // N·m, <0 = not specified
    double   max_velocity{-1.0};        // m/s, <0 = not specified
    double   thermal_load{0.0};         // K added to state.temperature
    double   power_required{0.0};       // W
    bool     is_reversible{true};
    int32_t  priority{0};
};

// ─────────────────────────────────────────────
//  Safety violation record
// ─────────────────────────────────────────────
enum class ViolationType : uint8_t {
    TORQUE_EXCEEDED       = 0,
    LOAD_FACTOR_EXCEEDED  = 1,
    TEMPERATURE_EXCEEDED  = 2,
    BATTERY_LOW           = 3,
    HAZARD_PROXIMITY      = 4,
    JOINT_LIMIT           = 5,
    COMM_BLACKOUT         = 6,
    STRUCTURAL_OVERLOAD   = 7,
    AXIOM_VIOLATION       = 8,
    UNKNOWN               = 255
};

struct Violation {
    ViolationType type{ViolationType::UNKNOWN};
    char          component[32]{};
    double        actual{0.0};
    double        limit{0.0};
};

struct SafetyCheckResult {
    bool      safe{false};
    Violation violations[MAX_VIOLATIONS]{};
    uint32_t  n_violations{0};
    double    confidence{0.0};

    void add_violation(ViolationType t, const char* comp,
                       double actual, double limit) {
        if (n_violations >= MAX_VIOLATIONS) return;
        violations[n_violations].type   = t;
        violations[n_violations].actual = actual;
        violations[n_violations].limit  = limit;
        std::strncpy(violations[n_violations].component, comp, 31);
        ++n_violations;
    }

    bool has_critical() const {
        for (uint32_t i = 0; i < n_violations; ++i) {
            auto t = violations[i].type;
            if (t == ViolationType::TORQUE_EXCEEDED   ||
                t == ViolationType::LOAD_FACTOR_EXCEEDED ||
                t == ViolationType::BATTERY_LOW        ||
                t == ViolationType::AXIOM_VIOLATION)
                return true;
        }
        return false;
    }
};

// ─────────────────────────────────────────────
//  Axiom system state (fed from ROS2 topic)
// ─────────────────────────────────────────────
struct AxiomSystemState {
    // Epistemic
    double   confidence{0.0};
    bool     uncertainty_explicit{false};
    double   uncertainty_current{1.0};
    double   uncertainty_max_credible{1.0};
    double   uncertainty_previous{1.0};
    bool     has_new_evidence{false};
    double   belief_oscillation_rate{0.0};
    double   belief_timestamp{0.0};
    double   belief_validity_window{1.0};
    // Authority
    uint32_t n_active_authorities{0};
    bool     authority_revoked_this_cycle{false};
    bool     authority_explicitly_defined{false};
    bool     is_acting{false};
    // Consequence
    char     evaluation_basis[32]{};         // "worst_case_credible"
    double   potential_harm{0.0};
    bool     action_is_reversible{true};
    double   distance_to_harm{1.0};
    double   constraint_tightness{1.0};
    // Refusal / Arbitration
    bool     refusal_is_illegal{false};
    bool     action_is_full_capability{false};
    bool     is_safe_for_full_capability{true};
    bool     is_optimizing{false};
    bool     is_safe{true};
    bool     all_actions_gated{true};
    bool     gate_decision_final{true};
    // Learning
    bool     learning_overrides{false};
    bool     is_irreversible_context{false};
    bool     online_learning_active{false};
    // Traceability
    bool     action_has_explanation{false};
    double   justification_timestamp{0.0};
    double   action_timestamp{0.0};
    // Mode
    SystemMode system_mode{SystemMode::NORMAL};
};

// ─────────────────────────────────────────────
//  Axiom validation result
// ─────────────────────────────────────────────
static constexpr uint32_t MAX_AXIOMS = 25;

struct AxiomResult {
    char     axiom_id[8]{};
    bool     passed{false};
    char     reason[128]{};
};

struct AxiomValidationResult {
    AxiomResult checks[MAX_AXIOMS]{};
    uint32_t    n_checked{0};
    bool        authority_valid{false};
    bool        action_permitted{false};
    double      timestamp{0.0};

    void add(const char* id, bool pass, const char* why) {
        if (n_checked >= MAX_AXIOMS) return;
        std::strncpy(checks[n_checked].axiom_id, id, 7);
        checks[n_checked].passed = pass;
        std::strncpy(checks[n_checked].reason, why, 127);
        ++n_checked;
    }
};

} // namespace arvs
