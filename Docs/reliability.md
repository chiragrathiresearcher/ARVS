# Space-Grade Reliability & Hardware Hardening

ARVS is engineered for mission-critical deployment in extreme environments (e.g., orbital platforms, planetary rovers, and high-radiation zones). This document details the low-level architectural features that ensure mathematical and temporal integrity even in the event of hardware-level interference.

---

## 1. Radiation-Hardened Mathematics (TMR)
In high-radiation environments, **Single Event Upsets (SEUs)** or "bit-flips" can corrupt variables in RAM. To combat this, the `RadiationHardenedMath` module (see `utils/math_utils.py`) implements **Triple Modular Redundancy (TMR)**.

### The TMR Process:
1.  **Redundant Paths**: Every critical calculation (such as a Dot Product or Kalman Gain) is performed three times using different memory orientations.
2.  **Voting Logic**: The system compares the three results. If one result deviates, it is discarded as a hardware anomaly.
3.  **Checksums**: Results are paired with SHA-256 or structural checksums to ensure they are not corrupted during transit between modules.



---

## 2. Temporal Integrity & Synchronization
Autonomous verification requires a "Single Source of Truth" for time. The `SpacecraftClock` (see `utils/timing.py`) provides high-precision synchronization:

* **Mission Elapsed Time (MET)**: Tracks time from the moment of activation, independent of local system drift.
* **Multi-Source Fusion**: Fuses time signals from GPS, NTP, and Internal Oscillators, using weighted averages to maintain stability if one source fails.
* **Watchdog Timers**: A "Pet-the-Dog" mechanism ensures that if the main ARVS loop hangs, the system automatically triggers a hardware reset to regain a safe state.

---

## 3. Physics-Based Validation
The `SpacecraftValidator` (see `utils/validation.py`) acts as a "Sanity Filter" for all data structures.

* **Range Constraints**: Ensures telemetry (e.g., Battery = 105% or Temp = -500K) is flagged as a sensor failure rather than processed as valid data.
* **Quaternion Normalization**: Automatically detects and corrects "drift" in orientation data caused by floating-point rounding errors over long missions.
* **Graceful Correction**: Minor anomalies are auto-corrected using historical averages, while major anomalies trigger a `ValidationError` and move the system to `DEGRADED` mode.



---

## 4. Stability in Estimation
The `KalmanFilterUtilities` provide specific protections for autonomous navigation:
* **Singularity Protection**: Prevents matrix inversion errors when sensor data is perfectly aligned or missing.
* **Positive Definite Enforcement**: Ensures covariance matrices stay mathematically valid, preventing the "divergence" crashes common in standard navigation filters.

---

## 5. Forensic Traceability
Every reliability event (a bit-flip detection, a watchdog pet, or a validation correction) is logged with a cryptographic hash. This allows mission controllers to reconstruct exactly how the system's "Immune System" responded to environmental stress.
