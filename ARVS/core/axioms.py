"""
ARVS Axiom Base - Closed, Non-Bypassable Safety Constitution
Version: 1.0 | Enforcement: Mandatory | Status: Active
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set, Tuple
import time
import logging

logger = logging.getLogger(__name__)


class AxiomCategory(Enum):
    """Categories of axioms."""
    EPISTEMIC = auto()       # Knowledge & Belief
    UNCERTAINTY = auto()     # Ignorance Handling
    AUTHORITY = auto()       # Decision Authority
    CONSEQUENCE = auto()     # Action & Consequence
    REFUSAL = auto()        # Refusal & Degradation
    ARBITRATION = auto()    # Arbitration & Enforcement
    LEARNING = auto()       # Learning & Adaptation
    TRACEABILITY = auto()   # Responsibility & Traceability
    CLOSURE = auto()        # System Closure


class AxiomSeverity(Enum):
    """Severity of axiom violation."""
    CRITICAL = auto()   # Immediate halt required
    HIGH = auto()       # Authority revocation
    MEDIUM = auto()     # Degraded mode required
    LOW = auto()        # Warning, continue with monitoring


@dataclass(frozen=True)
class Axiom:
    """
    Immutable axiom definition.
    Frozen to prevent runtime modification.
    """
    id: str                          # e.g., "E1", "A2", "Z"
    name: str                        # Short name
    category: AxiomCategory
    statement: str                   # Formal statement
    implication: str                 # Practical implication
    enforcement_condition: Optional[str] = None  # Stringified check
    severity: AxiomSeverity = AxiomSeverity.CRITICAL
    temporal_validity: Optional[float] = None  # Seconds, None = eternal
    dependencies: Set[str] = field(default_factory=set)  # Axioms this depends on


class ARVSAxiomBase:
    """
    Base class containing all ARVS axioms.
    This is the constitution of the system - non-negotiable.
    """
    
    # =========================================================================
    # I. EPISTEMIC AXIOMS (Knowledge & Belief)
    # =========================================================================
    E1 = Axiom(
        id="E1",
        name="No Omniscience",
        category=AxiomCategory.EPISTEMIC,
        statement="The system never assumes full knowledge of the world.",
        implication="All beliefs are provisional; perfect certainty is illegal unless explicitly proven.",
        enforcement_condition="not state.get('assumes_omniscience', False)",
        severity=AxiomSeverity.CRITICAL
    )
    
    E2 = Axiom(
        id="E2",
        name="Unknown ≡ High Uncertainty",
        category=AxiomCategory.EPISTEMIC,
        statement="Any unobserved, unmodeled, or unmeasurable factor is treated as maximum uncertainty.",
        implication="If it's outside the model → it's dangerous by default.",
        enforcement_condition="state.uncertainty.unmodeled_factor_present",
        severity=AxiomSeverity.HIGH
    )
    
    E3 = Axiom(
        id="E3",
        name="Belief Stability Requirement",
        category=AxiomCategory.EPISTEMIC,
        statement="Rapid belief oscillation invalidates authority, even if individual beliefs appear confident.",
        implication="Blocks classifier flickering, mode switching instability, last-millisecond optimism.",
        enforcement_condition="state.belief.oscillation_rate < MAX_OSCILLATION_THRESHOLD",
        severity=AxiomSeverity.HIGH
    )
    
    E4 = Axiom(
        id="E4",
        name="Stale Knowledge Invalidation",
        category=AxiomCategory.EPISTEMIC,
        statement="Beliefs exceeding their temporal validity window are invalid.",
        implication="Old data = wrong data.",
        enforcement_condition="current_time - state.belief.timestamp <= state.belief.validity_window",
        severity=AxiomSeverity.MEDIUM,
        temporal_validity=1.0  # Re-check every second
    )
    
    # =========================================================================
    # II. UNCERTAINTY AXIOMS (Ignorance Handling)
    # =========================================================================
    U1 = Axiom(
        id="U1",
        name="Explicit Uncertainty Mandate",
        category=AxiomCategory.UNCERTAINTY,
        statement="No decision may be made using implicit or hidden uncertainty.",
        implication="If uncertainty is not represented → decision is illegal.",
        enforcement_condition="state.uncertainty.is_explicit and state.uncertainty.is_quantified",
        severity=AxiomSeverity.CRITICAL
    )
    
    U2 = Axiom(
        id="U2",
        name="Uncertainty Aggregation Conservatism",
        category=AxiomCategory.UNCERTAINTY,
        statement="When uncertainties from multiple sources conflict, the maximum credible uncertainty dominates.",
        implication="No averaging your way to confidence.",
        enforcement_condition="state.uncertainty.aggregated == state.uncertainty.maximum_credible",
        severity=AxiomSeverity.HIGH
    )
    
    U3 = Axiom(
        id="U3",
        name="Uncertainty Monotonicity",
        category=AxiomCategory.UNCERTAINTY,
        statement="Uncertainty cannot decrease without new, independent evidence.",
        implication="Blocks wishful smoothing, confidence 'snapping back'.",
        enforcement_condition="state.uncertainty.current >= state.uncertainty.previous or state.has_new_evidence",
        severity=AxiomSeverity.HIGH
    )
    
    # =========================================================================
    # III. AUTHORITY AXIOMS (Who is allowed to decide)
    # =========================================================================
    A1 = Axiom(
        id="A1",
        name="Authority is Conditional",
        category=AxiomCategory.AUTHORITY,
        statement="Authority to act exists only while belief, uncertainty, and context satisfy all constraints.",
        implication="Authority is revocable at any instant.",
        enforcement_condition="state.authority.is_valid and state.authority.conditions_met",
        severity=AxiomSeverity.CRITICAL,
        dependencies={"E1", "E2", "U1", "U2", "U3"}
    )
    
    A2 = Axiom(
        id="A2",
        name="Authority is Singular",
        category=AxiomCategory.AUTHORITY,
        statement="At any moment, exactly one authority may issue commands.",
        implication="No competing planners, silent overrides, or race conditions.",
        enforcement_condition="len(state.active_authorities) == 1",
        severity=AxiomSeverity.CRITICAL
    )
    
    A3 = Axiom(
        id="A3",
        name="Authority Loss is Final per Cycle",
        category=AxiomCategory.AUTHORITY,
        statement="Once authority is revoked in a decision cycle, it cannot be re-granted without new evidence.",
        implication="Prevents 'panic re-enable'.",
        enforcement_condition="not state.authority.was_revoked_this_cycle or state.has_new_evidence",
        severity=AxiomSeverity.HIGH
    )
    
    A4 = Axiom(
        id="A4",
        name="No Implicit Authority",
        category=AxiomCategory.AUTHORITY,
        statement="If no authority is explicitly valid, the system defaults to no authority.",
        implication="Inaction is the baseline.",
        enforcement_condition="state.authority.is_explicitly_defined or not state.is_acting",
        severity=AxiomSeverity.HIGH
    )
    
    # =========================================================================
    # IV. ACTION & CONSEQUENCE AXIOMS
    # =========================================================================
    C1 = Axiom(
        id="C1",
        name="Consequence Precedence",
        category=AxiomCategory.CONSEQUENCE,
        statement="Actions are evaluated by worst-case credible consequence, not expected value.",
        implication="Explicitly rejects pure expected-reward logic.",
        enforcement_condition="state.evaluation.basis == 'worst_case_credible'",
        severity=AxiomSeverity.CRITICAL
    )
    
    C2 = Axiom(
        id="C2",
        name="Non-Linear Risk Scaling",
        category=AxiomCategory.CONSEQUENCE,
        statement="Required confidence increases super-linearly with potential harm.",
        implication="Small harm ≠ big harm × probability.",
        enforcement_condition="state.confidence_required >= RISK_SCALING_FN(state.potential_harm)",
        severity=AxiomSeverity.HIGH
    )
    
    C3 = Axiom(
        id="C3",
        name="Irreversibility Barrier",
        category=AxiomCategory.CONSEQUENCE,
        statement="Actions that cross irreversible state boundaries require stricter thresholds than reversible ones.",
        implication="Point-of-no-return logic is mandatory.",
        enforcement_condition="state.action.is_reversible or state.confidence >= IRREVERSIBLE_THRESHOLD",
        severity=AxiomSeverity.CRITICAL
    )
    
    C4 = Axiom(
        id="C4",
        name="Proximity Escalation",
        category=AxiomCategory.CONSEQUENCE,
        statement="As temporal or physical distance to harm decreases, constraints tighten automatically.",
        implication="Late decisions are conservative decisions.",
        enforcement_condition="state.constraints.tightness >= PROXIMITY_FN(state.distance_to_harm)",
        severity=AxiomSeverity.HIGH,
        temporal_validity=0.1  # Re-check frequently for proximity
    )
    
    # =========================================================================
    # V. REFUSAL & DEGRADATION AXIOMS
    # =========================================================================
    R1 = Axiom(
        id="R1",
        name="Refusal Legitimacy",
        category=AxiomCategory.REFUSAL,
        statement="Refusal, pause, degradation, or handover are always valid outcomes.",
        implication="Never treated as failure.",
        enforcement_condition="state.refusal_is_allowed or not state.has_failed",
        severity=AxiomSeverity.MEDIUM
    )
    
    R2 = Axiom(
        id="R2",
        name="Degradation over Action",
        category=AxiomCategory.REFUSAL,
        statement="When full action is unsafe, degraded modes must be preferred over normal operation.",
        implication="Graceful failure beats bold failure.",
        enforcement_condition="not state.action.is_full_capability or state.is_safe_for_full_capability",
        severity=AxiomSeverity.HIGH
    )
    
    R3 = Axiom(
        id="R3",
        name="No Forced Optimization",
        category=AxiomCategory.REFUSAL,
        statement="The system is never obligated to optimize performance under unsafe conditions.",
        implication="Kills 'we had to continue'.",
        enforcement_condition="not state.is_optimizing or state.is_safe",
        severity=AxiomSeverity.MEDIUM
    )
    
    # =========================================================================
    # VI. ARBITRATION & ENFORCEMENT AXIOMS
    # =========================================================================
    G1 = Axiom(
        id="G1",
        name="Single Mandatory Gate",
        category=AxiomCategory.ARBITRATION,
        statement="All actions must pass through a single arbitration gate.",
        implication="No side channels. No backdoors.",
        enforcement_condition="state.all_actions_passed_through_gate",
        severity=AxiomSeverity.CRITICAL
    )
    
    G2 = Axiom(
        id="G2",
        name="Gate Supremacy",
        category=AxiomCategory.ARBITRATION,
        statement="Arbitration outcomes override planners, learners, and heuristics without exception.",
        implication="If the gate says no, everything else is noise.",
        enforcement_condition="state.gate_decision.is_final",
        severity=AxiomSeverity.CRITICAL
    )
    
    G3 = Axiom(
        id="G3",
        name="Emergency is Not an Exception",
        category=AxiomCategory.ARBITRATION,
        statement="Emergency conditions increase conservatism; they do not relax rules.",
        implication="Blocks the most common excuse in accidents.",
        enforcement_condition="not state.is_emergency or state.constraints.tightness >= EMERGENCY_TIGHTNESS",
        severity=AxiomSeverity.CRITICAL
    )
    
    # =========================================================================
    # VII. LEARNING & ADAPTATION AXIOMS
    # =========================================================================
    L1 = Axiom(
        id="L1",
        name="Learning is Subordinate",
        category=AxiomCategory.LEARNING,
        statement="Learning systems may propose changes but cannot override safety axioms.",
        implication="Learning advises. Axioms decide.",
        enforcement_condition="not state.learning_overrides_axioms",
        severity=AxiomSeverity.CRITICAL
    )
    
    L2 = Axiom(
        id="L2",
        name="No Online Learning Across Irreversible Boundaries",
        category=AxiomCategory.LEARNING,
        statement="Learning may not modify behavior in irreversible contexts without validation.",
        implication="Stops 'learning on the edge'.",
        enforcement_condition="not (state.is_irreversible_context and state.is_online_learning_active)",
        severity=AxiomSeverity.CRITICAL
    )
    
    # =========================================================================
    # VIII. TRACEABILITY & RESPONSIBILITY AXIOMS
    # =========================================================================
    T1 = Axiom(
        id="T1",
        name="Decision Traceability",
        category=AxiomCategory.TRACEABILITY,
        statement="Every permitted action must be explainable in terms of belief, uncertainty, and axioms.",
        implication="If you can't explain it, it didn't happen.",
        enforcement_condition="state.action.has_explanation and state.explanation.references_axioms",
        severity=AxiomSeverity.HIGH
    )
    
    T2 = Axiom(
        id="T2",
        name="Post-Hoc Justification Invalidity",
        category=AxiomCategory.TRACEABILITY,
        statement="Decisions cannot be justified after execution; justification must exist beforehand.",
        implication="No retroactive excuses.",
        enforcement_condition="state.justification.timestamp <= state.action.timestamp",
        severity=AxiomSeverity.HIGH
    )
    
    # =========================================================================
    # IX. CLOSURE AXIOM (The Seal)
    # =========================================================================
    Z = Axiom(
        id="Z",
        name="No Undefined Behavior",
        category=AxiomCategory.CLOSURE,
        statement="Any situation not explicitly permitted by the axioms results in refusal or degradation.",
        implication="Kills: 'We didn't think of that case', 'The axioms didn't cover X, so we did Y'",
        enforcement_condition="state.is_explicitly_permitted or state.status == 'refusal'",
        severity=AxiomSeverity.CRITICAL,
        dependencies={axiom.id for axiom in [
            E1, E2, E3, E4, U1, U2, U3, A1, A2, A3, A4,
            C1, C2, C3, C4, R1, R2, R3, G1, G2, G3,
            L1, L2, T1, T2
        ]}  # Depends on ALL other axioms
    )
    
    # =========================================================================
    # COLLECTIONS AND UTILITIES
    # =========================================================================
    
    # All axioms in declaration order
    ALL_AXIOMS = [
        E1, E2, E3, E4,                    # Epistemic
        U1, U2, U3,                        # Uncertainty
        A1, A2, A3, A4,                    # Authority
        C1, C2, C3, C4,                    # Consequence
        R1, R2, R3,                        # Refusal
        G1, G2, G3,                        # Arbitration
        L1, L2,                            # Learning
        T1, T2,                            # Traceability
        Z                                   # Closure
    ]
    
    # Index by ID for fast lookup
    AXIOMS_BY_ID = {axiom.id: axiom for axiom in ALL_AXIOMS}
    
    # Axioms by category
    AXIOMS_BY_CATEGORY = {
        AxiomCategory.EPISTEMIC: [E1, E2, E3, E4],
        AxiomCategory.UNCERTAINTY: [U1, U2, U3],
        AxiomCategory.AUTHORITY: [A1, A2, A3, A4],
        AxiomCategory.CONSEQUENCE: [C1, C2, C3, C4],
        AxiomCategory.REFUSAL: [R1, R2, R3],
        AxiomCategory.ARBITRATION: [G1, G2, G3],
        AxiomCategory.LEARNING: [L1, L2],
        AxiomCategory.TRACEABILITY: [T1, T2],
        AxiomCategory.CLOSURE: [Z]
    }
    
    # Critical axioms that must always pass
    CRITICAL_AXIOMS = [axiom for axiom in ALL_AXIOMS 
                      if axiom.severity == AxiomSeverity.CRITICAL]
    
    @classmethod
    def get_axiom(cls, axiom_id: str) -> Axiom:
        """Get axiom by ID."""
        return cls.AXIOMS_BY_ID.get(axiom_id.upper())
    
    @classmethod
    def get_category_axioms(cls, category: AxiomCategory) -> List[Axiom]:
        """Get all axioms in a category."""
        return cls.AXIOMS_BY_CATEGORY.get(category, [])
    
    @classmethod
    def check_dependencies(cls, axiom_id: str, 
                          passed_axioms: Set[str]) -> bool:
        """
        Check if an axiom's dependencies are satisfied.
        
        Args:
            axiom_id: ID of axiom to check
            passed_axioms: Set of axiom IDs that have passed validation
            
        Returns:
            True if all dependencies are satisfied
        """
        axiom = cls.get_axiom(axiom_id)
        if not axiom:
            return False
        
        # Axiom Z depends on ALL other axioms
        if axiom_id == "Z":
            required = {a.id for a in cls.ALL_AXIOMS if a.id != "Z"}
            return required.issubset(passed_axioms)
        
        # Check regular dependencies
        return axiom.dependencies.issubset(passed_axioms)
    
    @classmethod
    def validate_axiom_set(cls) -> Dict[str, Any]:
        """
        Validate the entire axiom set for consistency.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "total_axioms": len(cls.ALL_AXIOMS),
            "critical_count": len(cls.CRITICAL_AXIOMS),
            "categories_covered": len(cls.AXIOMS_BY_CATEGORY),
            "closure_axiom_present": "Z" in cls.AXIOMS_BY_ID,
            "errors": [],
            "warnings": []
        }
        
        # Check for duplicate IDs
        ids = [axiom.id for axiom in cls.ALL_AXIOMS]
        if len(ids) != len(set(ids)):
            results["errors"].append("Duplicate axiom IDs found")
        
        # Check Z axiom dependencies
        z_axiom = cls.get_axiom("Z")
        if z_axiom:
            expected_deps = {axiom.id for axiom in cls.ALL_AXIOMS if axiom.id != "Z"}
            if z_axiom.dependencies != expected_deps:
                results["warnings"].append(
                    f"Axiom Z dependencies may be incomplete. "
                    f"Expected: {expected_deps}, Got: {z_axiom.dependencies}"
                )
        
        # Check temporal validity for epistemic axioms
        for axiom in cls.AXIOMS_BY_CATEGORY.get(AxiomCategory.EPISTEMIC, []):
            if axiom.temporal_validity is None and axiom.id != "E4":
                results["warnings"].append(
                    f"Epistemic axiom {axiom.id} has no temporal validity check"
                )
        
        return results
    
    @classmethod
    def generate_enforcement_code(cls, axiom_id: str) -> str:
        """
        Generate Python code for axiom enforcement.
        
        Args:
            axiom_id: ID of the axiom
            
        Returns:
            Python code snippet for enforcement
        """
        axiom = cls.get_axiom(axiom_id)
        if not axiom:
            return f"# Error: Axiom {axiom_id} not found"
        
        template = f'''
def enforce_{axiom.id}(state: SystemState) -> Tuple[bool, str]:
    """
    Enforce: {axiom.name}
    Statement: {axiom.statement}
    Implication: {axiom.implication}
    Severity: {axiom.severity.name}
    """
    try:
        # TODO: Implement actual enforcement logic
        # Current placeholder: {axiom.enforcement_condition or 'No condition specified'}
        
        # Example implementation structure:
        # if not ({axiom.enforcement_condition}):
        #     return False, f"Axiom {axiom.id} violated: {axiom.name}"
        
        # For now, assume compliance (REPLACE WITH ACTUAL LOGIC)
        return True, ""
        
    except Exception as e:
        logger.error(f"Failed to enforce {axiom.id}: {{e}}")
        # On enforcement failure, assume violation
        return False, f"Axiom {axiom.id} enforcement failed: {{str(e)}}"
'''
        return template.strip()


# =============================================================================
# AXIOM VALIDATOR CLASS
# =============================================================================

class AxiomValidator:
    """
    Runtime validator for ARVS axioms.
    """
    
    def __init__(self):
        self.violation_history = []
        self.last_validation_time = 0
        self.validation_interval = 0.1  # Seconds
        
    # ------------------------------------------------------------------
    # AXIOM ENFORCEMENT FUNCTIONS
    # Each function evaluates one axiom against real system_state fields.
    # system_state is expected to contain:
    #   'confidence'          float [0,1]   — current belief confidence
    #   'uncertainty_explicit' bool         — uncertainty is quantified
    #   'uncertainty_current'  float        — current uncertainty value
    #   'uncertainty_previous' float        — previous cycle uncertainty
    #   'has_new_evidence'    bool          — new sensor data this cycle
    #   'belief_oscillation_rate' float     — mode/belief flips per second
    #   'belief_timestamp'    float         — timestamp of current belief
    #   'belief_validity_window' float      — seconds before belief stale
    #   'active_authorities'  list          — list of active authority IDs
    #   'authority_revoked_this_cycle' bool — authority was revoked
    #   'is_acting'           bool          — system is currently actuating
    #   'evaluation_basis'    str           — 'worst_case_credible' or other
    #   'potential_harm'      float [0,1]   — severity of worst-case harm
    #   'action_is_reversible' bool         — can action be undone
    #   'distance_to_harm'    float         — metres or seconds to harm
    #   'system_mode'         str           — 'NORMAL','DEGRADED','EMERGENCY'
    #   'all_actions_gated'   bool          — all actions passed safety gate
    #   'gate_decision_final' bool          — gate output is not overridable
    #   'learning_overrides'  bool          — learner overrode an axiom
    #   'is_irreversible_context' bool      — current context is irreversible
    #   'online_learning_active' bool       — learner is updating live
    #   'action_has_explanation' bool       — decision has prior justification
    #   'justification_timestamp' float     — when justification was recorded
    #   'action_timestamp'    float         — when action was issued
    # ------------------------------------------------------------------

    MAX_OSCILLATION_THRESHOLD = 2.0   # flips/second before belief unstable
    IRREVERSIBLE_CONFIDENCE_THRESHOLD = 0.95
    EMERGENCY_TIGHTNESS = 1.2         # constraints must be tighter in emergency

    def _enforce_axiom(self, axiom_id: str, system_state: Dict[str, Any],
                       current_time: float) -> Tuple[bool, str]:
        """
        Evaluate a single axiom against real system_state.
        Returns (passes: bool, reason: str).
        """
        s = system_state  # alias for brevity

        # ---- EPISTEMIC ----
        if axiom_id == "E1":
            # System must not assume omniscience: confidence < 1.0
            confidence = s.get('confidence', 0.0)
            if confidence >= 1.0:
                return False, f"E1: confidence == {confidence:.3f} claims omniscience"
            return True, ""

        if axiom_id == "E2":
            # Unmodelled factors must be treated as high uncertainty
            uncertainty_explicit = s.get('uncertainty_explicit', False)
            if not uncertainty_explicit:
                return False, "E2: uncertainty is not explicitly quantified"
            return True, ""

        if axiom_id == "E3":
            # Belief must not oscillate rapidly
            rate = s.get('belief_oscillation_rate', 0.0)
            if rate > self.MAX_OSCILLATION_THRESHOLD:
                return False, (f"E3: belief oscillation rate {rate:.2f}/s "
                               f"exceeds threshold {self.MAX_OSCILLATION_THRESHOLD}")
            return True, ""

        if axiom_id == "E4":
            # Belief must not be stale
            belief_ts = s.get('belief_timestamp', 0.0)
            validity  = s.get('belief_validity_window', 1.0)
            age = current_time - belief_ts
            if age > validity:
                return False, f"E4: belief age {age:.2f}s exceeds window {validity:.2f}s"
            return True, ""

        # ---- UNCERTAINTY ----
        if axiom_id == "U1":
            explicit   = s.get('uncertainty_explicit', False)
            quantified = s.get('uncertainty_current') is not None
            if not (explicit and quantified):
                return False, "U1: decision uses implicit or unquantified uncertainty"
            return True, ""

        if axiom_id == "U2":
            # Aggregated uncertainty must equal maximum credible uncertainty
            current  = s.get('uncertainty_current', 0.0)
            max_cred = s.get('uncertainty_max_credible', current)
            if current < max_cred - 1e-6:
                return False, (f"U2: aggregated uncertainty {current:.4f} < "
                               f"max credible {max_cred:.4f}")
            return True, ""

        if axiom_id == "U3":
            # Uncertainty must not decrease without new evidence
            current  = s.get('uncertainty_current', 0.0)
            previous = s.get('uncertainty_previous', 0.0)
            new_ev   = s.get('has_new_evidence', False)
            if current < previous - 1e-6 and not new_ev:
                return False, (f"U3: uncertainty decreased {previous:.4f}→{current:.4f} "
                               f"without new evidence")
            return True, ""

        # ---- AUTHORITY ----
        if axiom_id == "A1":
            confidence = s.get('confidence', 0.0)
            explicit   = s.get('uncertainty_explicit', False)
            if confidence <= 0.0 or not explicit:
                return False, f"A1: authority conditions not met (conf={confidence:.3f})"
            return True, ""

        if axiom_id == "A2":
            active = s.get('active_authorities', [])
            if len(active) != 1:
                return False, f"A2: {len(active)} active authorities (must be exactly 1)"
            return True, ""

        if axiom_id == "A3":
            revoked   = s.get('authority_revoked_this_cycle', False)
            new_ev    = s.get('has_new_evidence', False)
            if revoked and not new_ev:
                return False, "A3: authority re-grant attempted without new evidence"
            return True, ""

        if axiom_id == "A4":
            is_acting  = s.get('is_acting', False)
            explicitly = s.get('authority_explicitly_defined', False)
            if is_acting and not explicitly:
                return False, "A4: system is acting without explicit authority definition"
            return True, ""

        # ---- CONSEQUENCE ----
        if axiom_id == "C1":
            basis = s.get('evaluation_basis', '')
            if basis != 'worst_case_credible':
                return False, f"C1: evaluation basis is '{basis}', must be 'worst_case_credible'"
            return True, ""

        if axiom_id == "C2":
            harm = s.get('potential_harm', 0.0)
            # Required confidence scales super-linearly with harm
            required_conf = min(1.0, harm ** 0.5 + 0.5 * harm)
            actual_conf   = s.get('confidence', 0.0)
            if actual_conf < required_conf - 1e-6:
                return False, (f"C2: confidence {actual_conf:.3f} < required {required_conf:.3f} "
                               f"for harm level {harm:.3f}")
            return True, ""

        if axiom_id == "C3":
            reversible = s.get('action_is_reversible', True)
            confidence = s.get('confidence', 0.0)
            if not reversible and confidence < self.IRREVERSIBLE_CONFIDENCE_THRESHOLD:
                return False, (f"C3: irreversible action with confidence {confidence:.3f} "
                               f"< threshold {self.IRREVERSIBLE_CONFIDENCE_THRESHOLD}")
            return True, ""

        if axiom_id == "C4":
            # Constraints must tighten as proximity to harm decreases
            # Represented as a normalised distance [0,1]; lower = closer
            distance = s.get('distance_to_harm', 1.0)
            # Tightness must be at least (1 - distance): closer = tighter
            required_tightness = max(0.0, 1.0 - distance)
            actual_tightness   = s.get('constraint_tightness', 1.0)
            if actual_tightness < required_tightness - 1e-6:
                return False, (f"C4: constraint tightness {actual_tightness:.3f} < "
                               f"required {required_tightness:.3f} at distance {distance:.3f}")
            return True, ""

        # ---- REFUSAL ----
        if axiom_id == "R1":
            # Refusal is always a valid outcome — this axiom cannot be violated
            # by a well-formed system; flag only if system has marked it illegal
            if s.get('refusal_is_illegal', False):
                return False, "R1: system has marked refusal as illegal — constitution violated"
            return True, ""

        if axiom_id == "R2":
            full_cap = s.get('action_is_full_capability', False)
            safe_fc  = s.get('is_safe_for_full_capability', True)
            if full_cap and not safe_fc:
                return False, "R2: full-capability action attempted under unsafe conditions"
            return True, ""

        if axiom_id == "R3":
            optimising = s.get('is_optimizing', False)
            safe       = s.get('is_safe', True)
            if optimising and not safe:
                return False, "R3: system is optimising performance under unsafe conditions"
            return True, ""

        # ---- ARBITRATION ----
        if axiom_id == "G1":
            gated = s.get('all_actions_gated', True)
            if not gated:
                return False, "G1: one or more actions bypassed the mandatory safety gate"
            return True, ""

        if axiom_id == "G2":
            final = s.get('gate_decision_final', True)
            if not final:
                return False, "G2: gate decision was overridden by a planner or learner"
            return True, ""

        if axiom_id == "G3":
            mode      = s.get('system_mode', 'NORMAL')
            tightness = s.get('constraint_tightness', 1.0)
            if mode == 'EMERGENCY' and tightness < self.EMERGENCY_TIGHTNESS:
                return False, (f"G3: emergency mode but constraint tightness {tightness:.3f} "
                               f"< required {self.EMERGENCY_TIGHTNESS}")
            return True, ""

        # ---- LEARNING ----
        if axiom_id == "L1":
            overrides = s.get('learning_overrides', False)
            if overrides:
                return False, "L1: learning system has overridden a safety axiom"
            return True, ""

        if axiom_id == "L2":
            irreversible = s.get('is_irreversible_context', False)
            learning     = s.get('online_learning_active', False)
            if irreversible and learning:
                return False, "L2: online learning is active in an irreversible context"
            return True, ""

        # ---- TRACEABILITY ----
        if axiom_id == "T1":
            has_explanation = s.get('action_has_explanation', False)
            if not has_explanation:
                return False, "T1: action has no prior explanation referencing axioms"
            return True, ""

        if axiom_id == "T2":
            just_ts   = s.get('justification_timestamp', current_time)
            action_ts = s.get('action_timestamp', current_time)
            if just_ts > action_ts + 1e-6:
                return False, (f"T2: justification timestamp {just_ts:.3f} is after "
                               f"action timestamp {action_ts:.3f}")
            return True, ""

        # ---- CLOSURE ----
        if axiom_id == "Z":
            # Handled separately in validate_state
            return True, ""

        return False, f"Unknown axiom id: {axiom_id}"

    def validate_state(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate current system state against all axioms.

        system_state must be a dict populated from real hardware telemetry.
        See _enforce_axiom docstring for required keys.

        Returns:
            Validation results dict with passed/failed lists and authority_valid flag.
        """
        current_time = time.time()

        if current_time - self.last_validation_time < self.validation_interval:
            return {"status": "rate_limited", "last_check": self.last_validation_time}

        self.last_validation_time = current_time

        results = {
            "timestamp": current_time,
            "total_checked": 0,
            "passed": [],
            "failed": [],
            "warnings": [],
            "authority_valid": True,
            "action_permitted": False
        }

        passed_axioms: set = set()

        # Evaluate all axioms except Z in declaration order
        for axiom in ARVSAxiomBase.ALL_AXIOMS:
            if axiom.id == "Z":
                continue

            results["total_checked"] += 1

            # Dependencies must pass first
            if not ARVSAxiomBase.check_dependencies(axiom.id, passed_axioms):
                results["failed"].append({
                    "axiom": axiom.id,
                    "reason": f"Unmet dependencies: {axiom.dependencies}",
                    "severity": axiom.severity.name
                })
                if axiom.severity == AxiomSeverity.CRITICAL:
                    results["authority_valid"] = False
                continue

            # Real enforcement against state
            passes, reason = self._enforce_axiom(axiom.id, system_state, current_time)

            if passes:
                results["passed"].append(axiom.id)
                passed_axioms.add(axiom.id)
            else:
                results["failed"].append({
                    "axiom": axiom.id,
                    "reason": reason,
                    "severity": axiom.severity.name
                })
                if axiom.severity == AxiomSeverity.CRITICAL:
                    results["authority_valid"] = False

                self.violation_history.append({
                    "timestamp": current_time,
                    "axiom": axiom.id,
                    "severity": axiom.severity.name,
                    "reason": reason,
                    "state_snapshot": {k: v for k, v in system_state.items()
                                       if not k.startswith('_')}
                })

        # Evaluate closure axiom Z last
        axiom_z = ARVSAxiomBase.Z
        results["total_checked"] += 1

        if ARVSAxiomBase.check_dependencies("Z", passed_axioms):
            if len(results["failed"]) == 0:
                results["passed"].append("Z")
                passed_axioms.add("Z")
            else:
                # Z still passes if system is correctly in refusal/degradation mode
                mode = system_state.get('system_mode', 'NORMAL')
                if mode in ('DEGRADED', 'SAFE_HOLD', 'EMERGENCY'):
                    results["passed"].append("Z")
                    passed_axioms.add("Z")
                    results["warnings"].append(
                        "Z: axiom failures present but system is in sanctioned degraded state"
                    )
                else:
                    results["failed"].append({
                        "axiom": "Z",
                        "reason": "Axiom failures exist and system is not in refusal/degradation",
                        "severity": axiom_z.severity.name
                    })
                    results["authority_valid"] = False
        else:
            results["failed"].append({
                "axiom": "Z",
                "reason": f"Dependencies not satisfied: {axiom_z.dependencies}",
                "severity": axiom_z.severity.name
            })
            results["authority_valid"] = False

        results["action_permitted"] = (
            results["authority_valid"] and
            len(results["failed"]) == 0 and
            "Z" in results["passed"]
        )

        return results
    
    def get_violation_report(self, limit: int = 10) -> List[Dict]:
        """Get recent violation history."""
        return self.violation_history[-limit:]


# =============================================================================
# EXPORTS AND MAIN GUARD
# =============================================================================

__all__ = [
    'AxiomCategory',
    'AxiomSeverity',
    'Axiom',
    'ARVSAxiomBase',
    'AxiomValidator'
]

if __name__ == "__main__":
    # Self-test when run directly
    print("=" * 60)
    print("ARVS AXIOM BASE - SELF TEST")
    print("=" * 60)
    
    validation = ARVSAxiomBase.validate_axiom_set()
    
    print(f"Total Axioms: {validation['total_axioms']}")
    print(f"Critical Axioms: {validation['critical_count']}")
    print(f"Categories: {validation['categories_covered']}")
    print(f"Closure Axiom Present: {validation['closure_axiom_present']}")
    
    if validation['errors']:
        print("\n❌ ERRORS:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    if not validation['errors']:
        print("\n✅ Axiom set is structurally valid.")
        
        # Show axiom summary
        print("\nAxiom Summary:")
        for category, axioms in ARVSAxiomBase.AXIOMS_BY_CATEGORY.items():
            print(f"\n{category.name}:")
            for axiom in axioms:
                print(f"  {axiom.id}: {axiom.name}")
    
    print("\n" + "=" * 60)
    print("Enforcement code sample for Axiom E1:")
    print("=" * 60)
    print(ARVSAxiomBase.generate_enforcement_code("E1"))

    