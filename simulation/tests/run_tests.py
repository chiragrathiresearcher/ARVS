"""
run_tests.py
ARVS Test Runner — unittest-based (no external pytest dependency)
Executes all test classes from test_arvs_full.py and reports results.

Usage:
  cd ARVS-main
  python simulation/tests/run_tests.py

Exit code 0 = all tests pass.  Exit code 1 = failures present.
"""

import sys
import os
import unittest
import time

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SIM  = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, SIM)

# ── Import test module and adapt pytest-style tests to unittest ───────────────
# pytest.approx → approximate equality check implemented inline

class approx:
    """Minimal pytest.approx replacement for unittest."""
    def __init__(self, expected, rel=1e-6, abs=None):
        self.expected = expected
        self.rel = rel
        self._abs = abs
    def __eq__(self, actual):
        if self._abs is not None:
            return builtins_abs(actual - self.expected) <= self._abs
        tol = self.rel * builtins_abs(self.expected) if self.expected != 0 else 1e-12
        return builtins_abs(actual - self.expected) <= tol
    def __repr__(self):
        return f"approx({self.expected})"

import builtins
builtins_abs = abs

# Patch pytest module so test_arvs_full.py can import it
import types
pytest_mock = types.ModuleType("pytest")
pytest_mock.approx = approx

class _RaisesCtx:
    def __init__(self, exc):
        self.exc = exc
    def __enter__(self): return self
    def __exit__(self, exctype, excval, tb):
        if exctype is None:
            raise AssertionError(f"Expected {self.exc} but no exception was raised")
        return issubclass(exctype, self.exc)

pytest_mock.raises = lambda exc: _RaisesCtx(exc)
sys.modules["pytest"] = pytest_mock

# ── Now import the test module ────────────────────────────────────────────────
from simulation.tests.test_arvs_full import (
    TestTelemetryBus,
    TestMockSensorDrivers,
    TestMDP,
    TestMissionPlanner,
    TestExperienceDB,
    TestSafetyGate,
    TestAxiomValidator,
    TestSimulationEngine,
    TestIntegration,
)


def make_suite():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestTelemetryBus,
        TestMockSensorDrivers,
        TestMDP,
        TestMissionPlanner,
        TestExperienceDB,
        TestSafetyGate,
        TestAxiomValidator,
        TestSimulationEngine,
        TestIntegration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    return suite


if __name__ == "__main__":
    print("=" * 72)
    print("ARVS TEST SUITE")
    print("=" * 72)
    t0     = time.time()
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(make_suite())
    elapsed = time.time() - t0

    print()
    print("=" * 72)
    print(f"Tests run    : {result.testsRun}")
    print(f"Failures     : {len(result.failures)}")
    print(f"Errors       : {len(result.errors)}")
    print(f"Skipped      : {len(result.skipped)}")
    print(f"Total time   : {elapsed:.1f}s")
    print(f"Result       : {'ALL PASS ✓' if result.wasSuccessful() else 'FAILURES ✗'}")
    print("=" * 72)

    sys.exit(0 if result.wasSuccessful() else 1)
