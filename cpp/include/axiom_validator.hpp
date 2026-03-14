/**
 * axiom_validator.hpp
 * ARVS C++ Axiom Validator — real-time enforcement of all 25 axioms
 *
 * Mirrors the Python AxiomValidator._enforce_axiom() logic exactly.
 * Called from the AxiomValidatorNode at 10 Hz (can be called faster).
 *
 * Enforcement contract:
 *   - Every CRITICAL axiom failure sets authority_valid = false immediately
 *   - Closure axiom Z is evaluated last
 *   - No heap, no exceptions — all errors reported via AxiomValidationResult
 */

#pragma once
#include "arvs_types.hpp"

namespace arvs {

class AxiomValidator {
public:
    AxiomValidator() = default;

    /**
     * validate — evaluate all 25 axioms against the supplied system state.
     * Thread-safe (read-only on state, writes only to result).
     */
    AxiomValidationResult validate(const AxiomSystemState& state,
                                   double current_time) const;

    /**
     * validate_single — evaluate one axiom by id ("E1", "G3", "Z", …).
     * Useful for targeted re-checks in the enforcement loop.
     */
    AxiomResult validate_single(const char* axiom_id,
                                const AxiomSystemState& state,
                                double current_time) const;

private:
    // One method per axiom — inline-able, no heap
    AxiomResult check_E1(const AxiomSystemState&) const;
    AxiomResult check_E2(const AxiomSystemState&) const;
    AxiomResult check_E3(const AxiomSystemState&) const;
    AxiomResult check_E4(const AxiomSystemState&, double t) const;
    AxiomResult check_U1(const AxiomSystemState&) const;
    AxiomResult check_U2(const AxiomSystemState&) const;
    AxiomResult check_U3(const AxiomSystemState&) const;
    AxiomResult check_A1(const AxiomSystemState&) const;
    AxiomResult check_A2(const AxiomSystemState&) const;
    AxiomResult check_A3(const AxiomSystemState&) const;
    AxiomResult check_A4(const AxiomSystemState&) const;
    AxiomResult check_C1(const AxiomSystemState&) const;
    AxiomResult check_C2(const AxiomSystemState&) const;
    AxiomResult check_C3(const AxiomSystemState&) const;
    AxiomResult check_C4(const AxiomSystemState&) const;
    AxiomResult check_R1(const AxiomSystemState&) const;
    AxiomResult check_R2(const AxiomSystemState&) const;
    AxiomResult check_R3(const AxiomSystemState&) const;
    AxiomResult check_G1(const AxiomSystemState&) const;
    AxiomResult check_G2(const AxiomSystemState&) const;
    AxiomResult check_G3(const AxiomSystemState&) const;
    AxiomResult check_L1(const AxiomSystemState&) const;
    AxiomResult check_L2(const AxiomSystemState&) const;
    AxiomResult check_T1(const AxiomSystemState&) const;
    AxiomResult check_T2(const AxiomSystemState&) const;
    AxiomResult check_Z (const AxiomValidationResult& partial,
                         const AxiomSystemState&) const;
};

} // namespace arvs
