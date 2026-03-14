/**
 * axiom_validator.cpp
 * ARVS Axiom Validator — implementation of all 25 axiom checks
 *
 * Logic mirrors Python AxiomValidator._enforce_axiom() exactly.
 * Kept in one file so the axiom logic is always in one auditable place.
 */

#include "axiom_validator.hpp"
#include <cstring>
#include <cstdio>
#include <cmath>

namespace arvs {

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

static inline AxiomResult pass(const char* id)
{
    AxiomResult r{};
    std::strncpy(r.axiom_id, id, 7);
    r.passed = true;
    return r;
}

static inline AxiomResult fail(const char* id, const char* reason)
{
    AxiomResult r{};
    std::strncpy(r.axiom_id, id, 7);
    r.passed = false;
    std::strncpy(r.reason, reason, 127);
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Top-level validate
// ─────────────────────────────────────────────────────────────────────────────

AxiomValidationResult AxiomValidator::validate(const AxiomSystemState& s,
                                                double t) const
{
    AxiomValidationResult result{};
    result.timestamp      = t;
    result.authority_valid = true;

    // Evaluate axioms in declaration order, stopping to mark authority_valid
    // false on any CRITICAL failure (all axioms are CRITICAL in this impl;
    // WARNING-level would need a severity table — add later if required).

    auto run = [&](AxiomResult ar) {
        result.add(ar.axiom_id, ar.passed, ar.reason);
        if (!ar.passed) result.authority_valid = false;
    };

    // ── Epistemic ────────────────────────────────────────────────────────────
    run(check_E1(s));
    run(check_E2(s));
    run(check_E3(s));
    run(check_E4(s, t));

    // ── Uncertainty ──────────────────────────────────────────────────────────
    run(check_U1(s));
    run(check_U2(s));
    run(check_U3(s));

    // ── Authority ────────────────────────────────────────────────────────────
    run(check_A1(s));
    run(check_A2(s));
    run(check_A3(s));
    run(check_A4(s));

    // ── Consequence ──────────────────────────────────────────────────────────
    run(check_C1(s));
    run(check_C2(s));
    run(check_C3(s));
    run(check_C4(s));

    // ── Refusal ──────────────────────────────────────────────────────────────
    run(check_R1(s));
    run(check_R2(s));
    run(check_R3(s));

    // ── Arbitration ──────────────────────────────────────────────────────────
    run(check_G1(s));
    run(check_G2(s));
    run(check_G3(s));

    // ── Learning ─────────────────────────────────────────────────────────────
    run(check_L1(s));
    run(check_L2(s));

    // ── Traceability ─────────────────────────────────────────────────────────
    run(check_T1(s));
    run(check_T2(s));

    // ── Closure (Z) — evaluated last ─────────────────────────────────────────
    run(check_Z(result, s));

    // action_permitted only when authority is valid AND Z passed
    bool z_passed = false;
    for (uint32_t i = 0; i < result.n_checked; ++i) {
        if (std::strncmp(result.checks[i].axiom_id, "Z", 1) == 0) {
            z_passed = result.checks[i].passed;
            break;
        }
    }
    result.action_permitted = result.authority_valid && z_passed;

    return result;
}

AxiomResult AxiomValidator::validate_single(const char* id,
                                             const AxiomSystemState& s,
                                             double t) const
{
    if (std::strncmp(id, "E1", 2) == 0) return check_E1(s);
    if (std::strncmp(id, "E2", 2) == 0) return check_E2(s);
    if (std::strncmp(id, "E3", 2) == 0) return check_E3(s);
    if (std::strncmp(id, "E4", 2) == 0) return check_E4(s, t);
    if (std::strncmp(id, "U1", 2) == 0) return check_U1(s);
    if (std::strncmp(id, "U2", 2) == 0) return check_U2(s);
    if (std::strncmp(id, "U3", 2) == 0) return check_U3(s);
    if (std::strncmp(id, "A1", 2) == 0) return check_A1(s);
    if (std::strncmp(id, "A2", 2) == 0) return check_A2(s);
    if (std::strncmp(id, "A3", 2) == 0) return check_A3(s);
    if (std::strncmp(id, "A4", 2) == 0) return check_A4(s);
    if (std::strncmp(id, "C1", 2) == 0) return check_C1(s);
    if (std::strncmp(id, "C2", 2) == 0) return check_C2(s);
    if (std::strncmp(id, "C3", 2) == 0) return check_C3(s);
    if (std::strncmp(id, "C4", 2) == 0) return check_C4(s);
    if (std::strncmp(id, "R1", 2) == 0) return check_R1(s);
    if (std::strncmp(id, "R2", 2) == 0) return check_R2(s);
    if (std::strncmp(id, "R3", 2) == 0) return check_R3(s);
    if (std::strncmp(id, "G1", 2) == 0) return check_G1(s);
    if (std::strncmp(id, "G2", 2) == 0) return check_G2(s);
    if (std::strncmp(id, "G3", 2) == 0) return check_G3(s);
    if (std::strncmp(id, "L1", 2) == 0) return check_L1(s);
    if (std::strncmp(id, "L2", 2) == 0) return check_L2(s);
    if (std::strncmp(id, "T1", 2) == 0) return check_T1(s);
    if (std::strncmp(id, "T2", 2) == 0) return check_T2(s);
    return fail(id, "unknown axiom id");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Axiom implementations
// ─────────────────────────────────────────────────────────────────────────────

// E1: system must not claim omniscience (confidence < 1.0)
AxiomResult AxiomValidator::check_E1(const AxiomSystemState& s) const {
    if (s.confidence >= 1.0 - 1e-9) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "E1: confidence=%.4f claims omniscience", s.confidence);
        return fail("E1", buf);
    }
    return pass("E1");
}

// E2: uncertainty must be explicitly quantified
AxiomResult AxiomValidator::check_E2(const AxiomSystemState& s) const {
    if (!s.uncertainty_explicit)
        return fail("E2", "E2: uncertainty is not explicitly quantified");
    return pass("E2");
}

// E3: belief must not oscillate rapidly
AxiomResult AxiomValidator::check_E3(const AxiomSystemState& s) const {
    if (s.belief_oscillation_rate > MAX_OSCILLATION_RATE) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "E3: oscillation=%.2f/s > threshold=%.2f/s",
            s.belief_oscillation_rate, MAX_OSCILLATION_RATE);
        return fail("E3", buf);
    }
    return pass("E3");
}

// E4: belief must not be stale
AxiomResult AxiomValidator::check_E4(const AxiomSystemState& s, double t) const {
    const double age = t - s.belief_timestamp;
    if (age > s.belief_validity_window) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "E4: belief age=%.2fs > window=%.2fs", age, s.belief_validity_window);
        return fail("E4", buf);
    }
    return pass("E4");
}

// U1: decision must use explicit quantified uncertainty
AxiomResult AxiomValidator::check_U1(const AxiomSystemState& s) const {
    if (!s.uncertainty_explicit)
        return fail("U1", "U1: decision uses implicit or unquantified uncertainty");
    return pass("U1");
}

// U2: aggregated uncertainty must equal maximum credible uncertainty
AxiomResult AxiomValidator::check_U2(const AxiomSystemState& s) const {
    if (s.uncertainty_current < s.uncertainty_max_credible - 1e-6) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "U2: current=%.4f < max_credible=%.4f",
            s.uncertainty_current, s.uncertainty_max_credible);
        return fail("U2", buf);
    }
    return pass("U2");
}

// U3: uncertainty must not decrease without new evidence
AxiomResult AxiomValidator::check_U3(const AxiomSystemState& s) const {
    if (s.uncertainty_current < s.uncertainty_previous - 1e-6 &&
        !s.has_new_evidence) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "U3: uncertainty decreased %.4f->%.4f without evidence",
            s.uncertainty_previous, s.uncertainty_current);
        return fail("U3", buf);
    }
    return pass("U3");
}

// A1: authority requires positive confidence with explicit uncertainty
AxiomResult AxiomValidator::check_A1(const AxiomSystemState& s) const {
    if (s.confidence <= 0.0 || !s.uncertainty_explicit) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "A1: authority conditions not met (conf=%.3f, explicit=%d)",
            s.confidence, (int)s.uncertainty_explicit);
        return fail("A1", buf);
    }
    return pass("A1");
}

// A2: exactly one active authority
AxiomResult AxiomValidator::check_A2(const AxiomSystemState& s) const {
    if (s.n_active_authorities != 1) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "A2: %u active authorities (must be exactly 1)",
            s.n_active_authorities);
        return fail("A2", buf);
    }
    return pass("A2");
}

// A3: authority re-grant requires new evidence
AxiomResult AxiomValidator::check_A3(const AxiomSystemState& s) const {
    if (s.authority_revoked_this_cycle && !s.has_new_evidence)
        return fail("A3", "A3: authority re-grant attempted without new evidence");
    return pass("A3");
}

// A4: acting requires explicitly defined authority
AxiomResult AxiomValidator::check_A4(const AxiomSystemState& s) const {
    if (s.is_acting && !s.authority_explicitly_defined)
        return fail("A4", "A4: acting without explicit authority definition");
    return pass("A4");
}

// C1: evaluation must use worst-case credible basis
AxiomResult AxiomValidator::check_C1(const AxiomSystemState& s) const {
    if (std::strncmp(s.evaluation_basis, "worst_case_credible", 19) != 0) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "C1: evaluation_basis='%s', must be 'worst_case_credible'",
            s.evaluation_basis);
        return fail("C1", buf);
    }
    return pass("C1");
}

// C2: confidence must scale super-linearly with potential harm
AxiomResult AxiomValidator::check_C2(const AxiomSystemState& s) const {
    const double required = std::fmin(1.0,
        std::sqrt(s.potential_harm) + 0.5 * s.potential_harm);
    if (s.confidence < required - 1e-6) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "C2: confidence=%.3f < required=%.3f for harm=%.3f",
            s.confidence, required, s.potential_harm);
        return fail("C2", buf);
    }
    return pass("C2");
}

// C3: irreversible actions require high confidence
AxiomResult AxiomValidator::check_C3(const AxiomSystemState& s) const {
    if (!s.action_is_reversible &&
        s.confidence < IRREVERSIBLE_CONF_THR - 1e-6) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "C3: irreversible action with confidence=%.3f < threshold=%.3f",
            s.confidence, IRREVERSIBLE_CONF_THR);
        return fail("C3", buf);
    }
    return pass("C3");
}

// C4: constraints tighten as proximity to harm decreases
AxiomResult AxiomValidator::check_C4(const AxiomSystemState& s) const {
    const double required = std::fmax(0.0, 1.0 - s.distance_to_harm);
    if (s.constraint_tightness < required - 1e-6) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "C4: tightness=%.3f < required=%.3f at distance=%.3f",
            s.constraint_tightness, required, s.distance_to_harm);
        return fail("C4", buf);
    }
    return pass("C4");
}

// R1: refusal must always be a legal outcome
AxiomResult AxiomValidator::check_R1(const AxiomSystemState& s) const {
    if (s.refusal_is_illegal)
        return fail("R1", "R1: system has marked refusal as illegal");
    return pass("R1");
}

// R2: full capability only when safe
AxiomResult AxiomValidator::check_R2(const AxiomSystemState& s) const {
    if (s.action_is_full_capability && !s.is_safe_for_full_capability)
        return fail("R2", "R2: full-capability action under unsafe conditions");
    return pass("R2");
}

// R3: optimisation only when safe
AxiomResult AxiomValidator::check_R3(const AxiomSystemState& s) const {
    if (s.is_optimizing && !s.is_safe)
        return fail("R3", "R3: optimising performance under unsafe conditions");
    return pass("R3");
}

// G1: all actions must be gated
AxiomResult AxiomValidator::check_G1(const AxiomSystemState& s) const {
    if (!s.all_actions_gated)
        return fail("G1", "G1: one or more actions bypassed the mandatory safety gate");
    return pass("G1");
}

// G2: gate decision must be final (non-overridable)
AxiomResult AxiomValidator::check_G2(const AxiomSystemState& s) const {
    if (!s.gate_decision_final)
        return fail("G2", "G2: gate decision was overridden by planner or learner");
    return pass("G2");
}

// G3: constraints tighten in emergency
AxiomResult AxiomValidator::check_G3(const AxiomSystemState& s) const {
    static constexpr double EMERGENCY_TIGHTNESS = 1.2;
    if (s.system_mode == SystemMode::EMERGENCY &&
        s.constraint_tightness < EMERGENCY_TIGHTNESS) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "G3: emergency mode but tightness=%.3f < required=%.3f",
            s.constraint_tightness, EMERGENCY_TIGHTNESS);
        return fail("G3", buf);
    }
    return pass("G3");
}

// L1: learning must not override safety axioms
AxiomResult AxiomValidator::check_L1(const AxiomSystemState& s) const {
    if (s.learning_overrides)
        return fail("L1", "L1: learning system has overridden a safety axiom");
    return pass("L1");
}

// L2: no online learning in irreversible contexts
AxiomResult AxiomValidator::check_L2(const AxiomSystemState& s) const {
    if (s.is_irreversible_context && s.online_learning_active)
        return fail("L2", "L2: online learning active in irreversible context");
    return pass("L2");
}

// T1: actions must have prior explanation
AxiomResult AxiomValidator::check_T1(const AxiomSystemState& s) const {
    if (!s.action_has_explanation)
        return fail("T1", "T1: action has no prior explanation referencing axioms");
    return pass("T1");
}

// T2: justification must precede action
AxiomResult AxiomValidator::check_T2(const AxiomSystemState& s) const {
    if (s.justification_timestamp > s.action_timestamp + 1e-6) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "T2: justification ts=%.3f is after action ts=%.3f",
            s.justification_timestamp, s.action_timestamp);
        return fail("T2", buf);
    }
    return pass("T2");
}

// Z: closure — all other axioms must pass, or system must be in degraded mode
AxiomResult AxiomValidator::check_Z(const AxiomValidationResult& partial,
                                     const AxiomSystemState& s) const
{
    // Count failures (excluding Z itself)
    uint32_t failures = 0;
    for (uint32_t i = 0; i < partial.n_checked; ++i) {
        if (!partial.checks[i].passed) ++failures;
    }

    if (failures == 0) return pass("Z");

    // Z still passes if system is correctly in a sanctioned degraded state
    if (s.system_mode == SystemMode::DEGRADED  ||
        s.system_mode == SystemMode::SAFE_HOLD  ||
        s.system_mode == SystemMode::EMERGENCY) {
        return pass("Z");   // caller logs "Z passed in degraded state"
    }

    char buf[128];
    std::snprintf(buf, sizeof(buf),
        "Z: %u axiom failures and system is not in refusal/degradation", failures);
    return fail("Z", buf);
}

} // namespace arvs
