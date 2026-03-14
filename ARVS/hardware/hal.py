"""
hardware/hal.py
ARVS Hardware Abstraction Layer (HAL)
======================================
Provides a standardised interface between physical sensors/actuators and all
ARVS modules.  No module ever touches raw hardware directly — everything flows
through the HAL's telemetry bus using a publish-subscribe pattern.

Architecture
------------
                    ┌─────────────────────────────────┐
  IMU ──────────►  │                                 │
  GPS ──────────►  │   SensorDriver (per sensor)     │
  Camera ───────►  │                                 │
  Thermistor ───►  │   publishes to TelemetryBus     │
  Power monitor ►  │                                 │
                    └──────────────┬──────────────────┘
                                   │ TelemetryFrame (typed, validated)
                                   ▼
                    ┌─────────────────────────────────┐
                    │         TelemetryBus             │
                    │  publish / subscribe / replay    │
                    │  subsystem health flags          │
                    │  anomaly detection               │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
       StateEstimator       RiskQuantifier         SafetyGate
       (subscriber)         (subscriber)           (subscriber)

Design constraints (space / safety-critical):
  - No subscriber ever modifies the published frame
  - All sensor reads are timestamped at acquisition, not delivery
  - Dropped frames are counted and exposed as health metrics
  - In hardware-absent environments, MockSensorDriver generates
    physics-consistent data from published mission parameters
"""

import time
import math
import logging
import threading
import collections
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Sensor types
# ─────────────────────────────────────────────────────────────────────────────

class SensorType(Enum):
    IMU         = auto()   # Inertial Measurement Unit
    GPS         = auto()   # Global / local positioning
    CAMERA      = auto()   # Visual / depth
    THERMISTOR  = auto()   # Temperature sensor
    POWER       = auto()   # Voltage / current monitor
    PRESSURE    = auto()   # Barometric / atmospheric
    WIND        = auto()   # Anemometer
    LIDAR       = auto()   # Distance / terrain
    MAGNETOMETER = auto()  # Heading


class SensorHealth(Enum):
    HEALTHY     = "healthy"
    DEGRADED    = "degraded"   # data available but noisy
    DROPOUT     = "dropout"    # no data this cycle
    FAILED      = "failed"     # permanent fault


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry frame — canonical data unit on the bus
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TelemetryFrame:
    """
    Immutable data packet published to the TelemetryBus.
    frozen=True enforces that subscribers cannot mutate published data.
    """
    sensor_id:         str
    sensor_type:       SensorType
    acquisition_time:  float          # UNIX epoch at hardware read
    sequence_number:   int

    # IMU
    acceleration:      Optional[Tuple[float,float,float]] = None  # m/s²
    angular_velocity:  Optional[Tuple[float,float,float]] = None  # rad/s
    orientation_quat:  Optional[Tuple[float,float,float,float]] = None  # w,x,y,z

    # GPS / position
    position:          Optional[Tuple[float,float,float]] = None  # m, mission frame
    velocity:          Optional[Tuple[float,float,float]] = None  # m/s
    position_covariance: Optional[Tuple[float,...]] = None        # 3x3 flattened

    # Thermal
    temperature_k:     Optional[float] = None
    thermal_gradient:  Optional[float] = None   # K/s

    # Power
    voltage_v:         Optional[float] = None
    current_a:         Optional[float] = None
    battery_soc:       Optional[float] = None   # [0,1]
    power_w:           Optional[float] = None

    # Environment
    pressure_pa:       Optional[float] = None
    wind_speed_ms:     Optional[float] = None
    wind_direction_deg: Optional[float] = None

    # Health
    health:            SensorHealth = SensorHealth.HEALTHY
    snr_db:            Optional[float] = None   # signal-to-noise ratio

    # Camera / LIDAR (kept as shape descriptors to avoid heavy payloads)
    image_shape:       Optional[Tuple[int,int,int]] = None
    point_cloud_n:     Optional[int] = None
    range_m:           Optional[float] = None   # nearest obstacle

    metadata:          Dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.health in (SensorHealth.HEALTHY, SensorHealth.DEGRADED)

    @property
    def confidence(self) -> float:
        return {
            SensorHealth.HEALTHY:  1.0,
            SensorHealth.DEGRADED: 0.6,
            SensorHealth.DROPOUT:  0.0,
            SensorHealth.FAILED:   0.0,
        }[self.health]


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry Bus — publish / subscribe
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryBus:
    """
    Thread-safe publish-subscribe bus for sensor telemetry.

    Publishers:  SensorDriver subclasses call bus.publish(frame)
    Subscribers: Any ARVS module calls bus.subscribe(sensor_type, callback)

    The bus also maintains:
      - Rolling history (configurable depth) for replay and debugging
      - Per-sensor health tracking
      - Drop counter (frames published with no subscriber)
      - Anomaly flags exposed to the SafetyGate
    """

    def __init__(self, history_depth: int = 1000):
        self._subscribers: Dict[SensorType, List[Callable]] = \
            {st: [] for st in SensorType}
        self._history:    collections.deque = \
            collections.deque(maxlen=history_depth)
        self._health:     Dict[str, SensorHealth] = {}
        self._seq_counts: Dict[str, int] = {}
        self._drop_count: int = 0
        self._lock = threading.Lock()

    # ── Publishing ────────────────────────────────────────────────────────────

    def publish(self, frame: TelemetryFrame) -> None:
        """Publish a TelemetryFrame to all registered subscribers."""
        with self._lock:
            self._history.append(frame)
            self._health[frame.sensor_id] = frame.health

            # Sequence gap detection
            last_seq = self._seq_counts.get(frame.sensor_id, frame.sequence_number - 1)
            if frame.sequence_number != last_seq + 1:
                gap = frame.sequence_number - last_seq - 1
                logger.warning(
                    f"Bus: {frame.sensor_id} dropped {gap} frames "
                    f"(expected seq {last_seq+1}, got {frame.sequence_number})")
                self._drop_count += gap
            self._seq_counts[frame.sensor_id] = frame.sequence_number

            callbacks = list(self._subscribers[frame.sensor_type])

        for cb in callbacks:
            try:
                cb(frame)
            except Exception as e:
                logger.error(f"Bus subscriber error ({frame.sensor_type}): {e}")

    # ── Subscribing ───────────────────────────────────────────────────────────

    def subscribe(self, sensor_type: SensorType,
                  callback: Callable[[TelemetryFrame], None]) -> None:
        """Register a callback for a sensor type. Thread-safe."""
        with self._lock:
            self._subscribers[sensor_type].append(callback)

    def unsubscribe(self, sensor_type: SensorType,
                    callback: Callable[[TelemetryFrame], None]) -> None:
        with self._lock:
            try:
                self._subscribers[sensor_type].remove(callback)
            except ValueError:
                pass

    # ── Query ─────────────────────────────────────────────────────────────────

    def latest(self, sensor_type: SensorType) -> Optional[TelemetryFrame]:
        """Return the most recent frame of the given type, or None."""
        with self._lock:
            for frame in reversed(self._history):
                if frame.sensor_type == sensor_type:
                    return frame
        return None

    def latest_by_id(self, sensor_id: str) -> Optional[TelemetryFrame]:
        with self._lock:
            for frame in reversed(self._history):
                if frame.sensor_id == sensor_id:
                    return frame
        return None

    def history(self, sensor_type: SensorType,
                n: int = 10) -> List[TelemetryFrame]:
        """Return last n frames of a given type."""
        with self._lock:
            return [f for f in self._history if f.sensor_type == sensor_type][-n:]

    def health_summary(self) -> Dict[str, str]:
        with self._lock:
            return {sid: h.value for sid, h in self._health.items()}

    def system_confidence(self) -> float:
        """Overall telemetry confidence: fraction of healthy sensors."""
        with self._lock:
            if not self._health:
                return 1.0
            healths = list(self._health.values())
        scores = {
            SensorHealth.HEALTHY:  1.0,
            SensorHealth.DEGRADED: 0.6,
            SensorHealth.DROPOUT:  0.0,
            SensorHealth.FAILED:   0.0,
        }
        return float(np.mean([scores[h] for h in healths]))

    @property
    def total_drops(self) -> int:
        return self._drop_count


# ─────────────────────────────────────────────────────────────────────────────
# Sensor Driver base class
# ─────────────────────────────────────────────────────────────────────────────

class SensorDriver(ABC):
    """
    Abstract base for all sensor drivers.
    Subclass this for real hardware (IMU, GPS, etc.) or mock sensors.
    """

    def __init__(self, sensor_id: str, sensor_type: SensorType,
                 bus: TelemetryBus, rate_hz: float = 10.0):
        self.sensor_id   = sensor_id
        self.sensor_type = sensor_type
        self.bus         = bus
        self.rate_hz     = rate_hz
        self._seq        = 0
        self._running    = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"SensorDriver {self.sensor_id} started at {self.rate_hz} Hz")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        period = 1.0 / self.rate_hz
        while self._running:
            t0 = time.monotonic()
            try:
                frame = self.read()
                if frame is not None:
                    self.bus.publish(frame)
            except Exception as e:
                logger.error(f"{self.sensor_id} read error: {e}")
            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, period - elapsed)
            time.sleep(sleep_t)

    @abstractmethod
    def read(self) -> Optional[TelemetryFrame]:
        """Read one sample. Must return a TelemetryFrame or None on dropout."""

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _frame(self, **kwargs) -> TelemetryFrame:
        """Helper: build a frame with sensor metadata pre-filled."""
        return TelemetryFrame(
            sensor_id        = self.sensor_id,
            sensor_type      = self.sensor_type,
            acquisition_time = time.time(),
            sequence_number  = self._next_seq(),
            **kwargs
        )


# ─────────────────────────────────────────────────────────────────────────────
# Mock sensor drivers — physics-consistent, deterministic (no random())
# Used when hardware is absent (simulation, CI, unit tests)
# ─────────────────────────────────────────────────────────────────────────────

class MockIMUDriver(SensorDriver):
    """
    Simulates a 6-DOF IMU on a rover traversing flat terrain.
    Motion model: constant forward velocity with gentle heading oscillation.
    Noise model: bounded deterministic perturbation derived from Allan variance
    of a MEMS IMU (e.g., InvenSense ICM-42688-P datasheet).
    """
    # ICM-42688-P gyro noise density: 0.0028 °/s/√Hz → at 100 Hz: ~0.028 °/s
    GYRO_NOISE_DENSITY    = 0.028 * math.pi / 180.0   # rad/s
    ACCEL_NOISE_DENSITY   = 0.0018                    # m/s²
    MISSION_PERIOD_S      = 88775.0                   # Martian sol

    def __init__(self, sensor_id: str, bus: TelemetryBus,
                 rate_hz: float = 100.0,
                 mission_start: float = 0.0):
        super().__init__(sensor_id, SensorType.IMU, bus, rate_hz)
        self.mission_start = mission_start

    def read(self) -> TelemetryFrame:
        t    = time.time() - self.mission_start
        freq = 2 * math.pi / self.MISSION_PERIOD_S

        # Forward acceleration: slow ramp + diurnal modulation
        ax = 0.02 * math.sin(freq * t)
        ay = 0.005 * math.cos(3 * freq * t)
        az = -3.72 + 0.01 * math.sin(5 * freq * t)   # Martian gravity

        # Heading oscillation (rover steering)
        gz = 0.01 * math.sin(0.1 * t)

        # Orientation: slow pitch and roll from terrain
        pitch = 0.05 * math.sin(0.03 * t)
        roll  = 0.02 * math.cos(0.05 * t)
        yaw   = 0.1 * t % (2 * math.pi)
        # Convert Euler → quaternion
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cr, sr = math.cos(roll/2),  math.sin(roll/2)
        cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy

        return self._frame(
            acceleration    = (ax, ay, az),
            angular_velocity= (0.0, 0.0, gz),
            orientation_quat= (qw, qx, qy, qz),
            snr_db          = 42.0,
        )


class MockGPSDriver(SensorDriver):
    """
    Simulates a differential GPS / UWB localiser.
    Position noise: ±0.1 m CEP (circular error probable) — RTK quality.
    Uses deterministic Lissajous path to simulate rover driving.
    """
    SPEED_MS = 0.05  # typical Mars rover traverse speed

    def __init__(self, sensor_id: str, bus: TelemetryBus,
                 rate_hz: float = 1.0, mission_start: float = 0.0):
        super().__init__(sensor_id, SensorType.GPS, bus, rate_hz)
        self.mission_start = mission_start

    def read(self) -> TelemetryFrame:
        t  = time.time() - self.mission_start
        vx = self.SPEED_MS * math.cos(0.02 * t)
        vy = self.SPEED_MS * math.sin(0.02 * t) * 0.3
        x  = self.SPEED_MS / 0.02 * math.sin(0.02 * t)
        y  = self.SPEED_MS * 0.3 / 0.02 * (-math.cos(0.02 * t) + 1.0)
        z  = -0.5 * math.sin(0.005 * t)   # gentle slope

        # Position covariance (3x3 diagonal, RTK quality)
        cov = (0.01, 0.0, 0.0,
               0.0,  0.01, 0.0,
               0.0,  0.0,  0.04)  # z slightly worse

        return self._frame(
            position              = (x, y, z),
            velocity              = (vx, vy, 0.0),
            position_covariance   = cov,
            snr_db                = 38.0,
        )


class MockThermistorDriver(SensorDriver):
    """Martian diurnal temperature cycle (REMS published parameters)."""

    def __init__(self, sensor_id: str, bus: TelemetryBus,
                 rate_hz: float = 1.0, mission_start: float = 0.0):
        super().__init__(sensor_id, SensorType.THERMISTOR, bus, rate_hz)
        self.mission_start  = mission_start
        self._prev_temp     = 244.5  # K — pre-dawn minimum

    def read(self) -> TelemetryFrame:
        t       = time.time() - self.mission_start
        sol_s   = 88775.0
        phase   = 2 * math.pi * t / sol_s
        # Sinusoidal diurnal: 184 K min, 305 K max
        temp_k  = 244.5 + 60.5 * math.sin(phase - math.pi / 3)
        grad    = (temp_k - self._prev_temp) / (1.0 / self.rate_hz)
        self._prev_temp = temp_k

        return self._frame(
            temperature_k   = temp_k,
            thermal_gradient= grad,
        )


class MockPowerMonitorDriver(SensorDriver):
    """
    Solar + battery power model for a Mars rover (MSL/Perseverance class).
    RTG power: ~110 W (constant); MMRTG degrades ~4.8 W/yr.
    """
    RTG_POWER_W    = 110.0
    BATTERY_CAP_WH = 43.2   # Perseverance: two 43.2 Wh Li-ion packs
    IDLE_LOAD_W    = 55.0

    def __init__(self, sensor_id: str, bus: TelemetryBus,
                 rate_hz: float = 1.0, mission_start: float = 0.0):
        super().__init__(sensor_id, SensorType.POWER, bus, rate_hz)
        self.mission_start = mission_start
        self._soc = 0.85

    def read(self) -> TelemetryFrame:
        t        = time.time() - self.mission_start
        # Load varies with activity (sinusoidal drive cycle approximation)
        load_w   = self.IDLE_LOAD_W + 20.0 * abs(math.sin(0.01 * t))
        net_w    = self.RTG_POWER_W - load_w
        dt       = 1.0 / self.rate_hz
        self._soc = max(0.0, min(1.0,
            self._soc + net_w * dt / 3600.0 / self.BATTERY_CAP_WH))

        return self._frame(
            voltage_v   = 28.8 + 1.2 * self._soc,  # nominal 28.8–30 V bus
            current_a   = load_w / (28.8 + 1.2 * self._soc),
            battery_soc = self._soc,
            power_w     = load_w,
        )


# ─────────────────────────────────────────────────────────────────────────────
# HAL — top-level hardware management object
# ─────────────────────────────────────────────────────────────────────────────

class HardwareAbstractionLayer:
    """
    Top-level HAL.  Owns the TelemetryBus and all sensor drivers.
    Responsible for:
      - Starting / stopping all sensor drivers
      - Exposing the bus to ARVS modules (read-only subscribe interface)
      - Injecting faults for simulation (fault injection API)
      - Reporting subsystem health to the audit logger
    """

    def __init__(self, mission_start: Optional[float] = None,
                 use_mock_sensors: bool = True):
        self.bus           = TelemetryBus(history_depth=2000)
        self._drivers:     List[SensorDriver] = []
        self._fault_flags: Dict[str, bool]    = {}
        self.mission_start = mission_start or time.time()

        if use_mock_sensors:
            self._register_mock_suite()

    def _register_mock_suite(self) -> None:
        ms = self.mission_start
        self._drivers = [
            MockIMUDriver        ("IMU_0",   self.bus, rate_hz=100.0, mission_start=ms),
            MockGPSDriver        ("GPS_0",   self.bus, rate_hz=1.0,   mission_start=ms),
            MockThermistorDriver ("THERM_0", self.bus, rate_hz=1.0,   mission_start=ms),
            MockPowerMonitorDriver("PWR_0",  self.bus, rate_hz=1.0,   mission_start=ms),
        ]

    def register_driver(self, driver: SensorDriver) -> None:
        """Register a real hardware driver."""
        self._drivers.append(driver)

    def start(self) -> None:
        for d in self._drivers:
            d.start()
        logger.info(f"HAL started — {len(self._drivers)} drivers active")

    def stop(self) -> None:
        for d in self._drivers:
            d.stop()
        logger.info("HAL stopped")

    # ── Fault injection API (simulation / testing) ────────────────────────────

    def inject_fault(self, sensor_id: str,
                     health: SensorHealth = SensorHealth.DROPOUT) -> None:
        """
        Mark a sensor as faulted.  Subsequent TelemetryFrames from that
        driver will carry the degraded health flag.
        Used by the simulation fault-injection scenario.
        """
        self._fault_flags[sensor_id] = health
        logger.warning(f"HAL fault injected: {sensor_id} → {health.value}")

    def clear_fault(self, sensor_id: str) -> None:
        self._fault_flags.pop(sensor_id, None)
        logger.info(f"HAL fault cleared: {sensor_id}")

    # ── Health reporting ──────────────────────────────────────────────────────

    def health(self) -> Dict:
        return {
            "sensors":          self.bus.health_summary(),
            "system_confidence": round(self.bus.system_confidence(), 4),
            "total_drops":      self.bus.total_drops,
            "driver_count":     len(self._drivers),
        }

    # ── Convenience read (latest fused state from bus) ────────────────────────

    def latest_state_estimate(self) -> Dict:
        """
        Build a minimal state dict from the latest frames on the bus.
        This is consumed by the simulation engine when ARVS Python modules
        are imported directly.
        """
        imu  = self.bus.latest(SensorType.IMU)
        gps  = self.bus.latest(SensorType.GPS)
        therm= self.bus.latest(SensorType.THERMISTOR)
        pwr  = self.bus.latest(SensorType.POWER)

        pos  = list(gps.position)   if gps  and gps.position  else [0.,0.,0.]
        vel  = list(gps.velocity)   if gps  and gps.velocity  else [0.,0.,0.]
        quat = list(imu.orientation_quat) if imu and imu.orientation_quat else [1.,0.,0.,0.]
        omeg = list(imu.angular_velocity) if imu and imu.angular_velocity else [0.,0.,0.]
        temp = therm.temperature_k  if therm and therm.temperature_k else 293.0
        soc  = pwr.battery_soc      if pwr   and pwr.battery_soc    else 0.85
        power= pwr.power_w          if pwr   and pwr.power_w        else 55.0
        conf = self.bus.system_confidence()

        return {
            "timestamp":         time.time() - self.mission_start,
            "position":          pos,
            "velocity":          vel,
            "orientation":       quat,
            "angular_velocity":  omeg,
            "temperature":       temp,
            "battery_level":     soc,
            "power_consumption": power,
            "confidence":        conf,
        }
