"""
telemetry_loader.py
ARVS Simulation — Real Telemetry Data Loader

Sources:
  1. NASA REMS (Mars Science Laboratory) — temperature, pressure, UV, wind
     PDS Atmospheres Node: https://pds-atmospheres.nmsu.edu/PDS/data/mslrem_1001/
  2. NASA SPICE kernels — spacecraft attitude, position, time (via spiceypy if available,
     otherwise uses pre-computed geometry from HORIZONS JSON API)
  3. ESA Rosetta / Mars Express housekeeping — PSA archive CSV
     https://archives.esac.esa.int/psa/
  4. ISS OSDR — NASA Open Science Data Repository
     https://osdr.nasa.gov/bio/repo/

Network unavailability is handled gracefully: each loader falls back to a
physics-consistent synthetic dataset derived from published mission parameters.
All synthetic fallbacks are clearly labelled in the returned metadata dict.

Returned telemetry is normalised to a common TelemetryFrame dataclass so the
simulation engine is data-source agnostic.
"""

import os
import io
import csv
import json
import math
import time
import logging
import hashlib
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Common telemetry frame
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TelemetryFrame:
    """
    One timestep of spacecraft/rover telemetry.
    All fields use SI units unless noted.
    """
    # Provenance
    source: str                    # "REMS", "SPICE", "ESA_ROSETTA", "ISS_OSDR", "SYNTHETIC_*"
    mission_time_s: float          # seconds from mission epoch
    utc_timestamp: str             # ISO-8601 if available, else empty

    # Position & attitude (mission frame)
    position_m: np.ndarray         # [x, y, z]
    velocity_ms: np.ndarray        # [vx, vy, vz]
    orientation_quat: np.ndarray   # [w, x, y, z]
    angular_velocity_rads: np.ndarray  # [wx, wy, wz]

    # Thermal
    temperature_k: float           # chassis / environment temperature
    thermal_gradient_k_per_s: float

    # Power
    battery_level: float           # [0, 1]
    power_consumption_w: float
    solar_irradiance_w_m2: float

    # Environment (mission-specific)
    ambient_pressure_pa: float
    wind_speed_ms: float
    uv_index: float

    # Health
    confidence: float              # [0, 1] — degrades with sensor dropout
    sensor_flags: Dict[str, bool]  # True = healthy

    # Extras
    metadata: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# 1. NASA REMS loader
# ─────────────────────────────────────────────────────────────────────────────

# Public REMS RDR browse data (no auth required, HTTP)
REMS_BASE = (
    "https://pds-atmospheres.nmsu.edu/PDS/data/mslrem_1001/DATA/SOL00001_00089/"
    "REMS_SOL00001_SOL00001_RDR.TAB"
)

REMS_FALLBACK_PARAMS = {
    # Published MSL mission statistics (Gómez-Elvira et al. 2012)
    "temp_surface_min_k":  184.0,
    "temp_surface_max_k":  305.0,
    "pressure_min_pa":     730.0,
    "pressure_max_pa":     900.0,
    "wind_speed_max_ms":   22.0,
    "sol_duration_s":      88775.0,  # Martian sol
}


def load_rems(n_frames: int = 200) -> List[TelemetryFrame]:
    """
    Load MSL REMS RDR telemetry.
    Falls back to physics-consistent synthetic Martian diurnal cycle.
    """
    try:
        logger.info("Fetching REMS data from NASA PDS...")
        req = urllib.request.Request(REMS_BASE, headers={"User-Agent": "ARVS-Sim/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("latin-1")
        frames = _parse_rems_tab(raw, n_frames)
        logger.info(f"REMS: loaded {len(frames)} real frames")
        return frames
    except Exception as e:
        logger.warning(f"REMS fetch failed ({e}), using synthetic Martian diurnal cycle")
        return _synthetic_rems(n_frames)


def _parse_rems_tab(raw: str, n_frames: int) -> List[TelemetryFrame]:
    frames = []
    lines = [l for l in raw.splitlines() if l.strip() and not l.startswith("#")]
    p = REMS_FALLBACK_PARAMS
    for i, line in enumerate(lines[:n_frames]):
        cols = line.split(",")
        if len(cols) < 6:
            continue
        try:
            t_s     = float(cols[0]) if cols[0].strip() else i * 1.0
            temp_k  = float(cols[2]) if cols[2].strip() else 240.0
            pres_pa = float(cols[3]) if cols[3].strip() else 800.0
            wind    = float(cols[4]) if cols[4].strip() else 3.0
        except ValueError:
            continue

        frames.append(TelemetryFrame(
            source="REMS",
            mission_time_s=t_s,
            utc_timestamp=cols[1].strip() if len(cols) > 1 else "",
            position_m=np.array([t_s * 0.01, 0.0, 0.0]),
            velocity_ms=np.array([0.01, 0.0, 0.0]),
            orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity_rads=np.array([0.0, 0.0, 0.0]),
            temperature_k=temp_k,
            thermal_gradient_k_per_s=0.0,
            battery_level=max(0.2, 1.0 - t_s / (p["sol_duration_s"] * 2)),
            power_consumption_w=55.0,
            solar_irradiance_w_m2=590.0 * max(0, math.sin(math.pi * t_s / p["sol_duration_s"])),
            ambient_pressure_pa=pres_pa,
            wind_speed_ms=wind,
            uv_index=2.0,
            confidence=0.95,
            sensor_flags={"temp": True, "pressure": True, "wind": True},
            metadata={"sol": 1, "real_data": True}
        ))
    return frames


def _synthetic_rems(n_frames: int) -> List[TelemetryFrame]:
    """Martian diurnal cycle derived from REMS published statistics."""
    p = REMS_FALLBACK_PARAMS
    frames = []
    dt = p["sol_duration_s"] / n_frames

    for i in range(n_frames):
        t = i * dt
        phase = 2 * math.pi * t / p["sol_duration_s"]

        # Temperature: sinusoidal with pre-dawn minimum
        temp_k = (p["temp_surface_min_k"] + p["temp_surface_max_k"]) / 2 + \
                 ((p["temp_surface_max_k"] - p["temp_surface_min_k"]) / 2) * \
                 math.sin(phase - math.pi / 3)

        pressure_pa = p["pressure_min_pa"] + \
                      (p["pressure_max_pa"] - p["pressure_min_pa"]) * \
                      0.5 * (1 + math.sin(phase))

        wind_ms = max(0.0, p["wind_speed_max_ms"] * 0.3 *
                      (1 + 0.5 * math.sin(phase + math.pi)))

        solar = max(0.0, 590.0 * math.sin(math.pi * t / p["sol_duration_s"]))
        battery = max(0.15, 0.95 - 0.4 * (1 - solar / 590.0) * (t / p["sol_duration_s"]))

        frames.append(TelemetryFrame(
            source="SYNTHETIC_REMS",
            mission_time_s=t,
            utc_timestamp="",
            position_m=np.array([t * 0.01, 0.0, 0.0]),
            velocity_ms=np.array([0.01, 0.0, 0.0]),
            orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity_rads=np.array([0.0, 0.0, 0.001 * math.sin(phase)]),
            temperature_k=temp_k,
            thermal_gradient_k_per_s=(temp_k - frames[-1].temperature_k) / dt
                                      if frames else 0.0,
            battery_level=battery,
            power_consumption_w=55.0 + 15.0 * (1 - solar / 600.0),
            solar_irradiance_w_m2=solar,
            ambient_pressure_pa=pressure_pa,
            wind_speed_ms=wind_ms,
            uv_index=max(0.0, 2.5 * math.sin(phase)),
            confidence=0.92,
            sensor_flags={"temp": True, "pressure": True, "wind": True, "uv": True},
            metadata={"sol": 1, "real_data": False, "model": "REMS_diurnal_stats"}
        ))
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# 2. NASA SPICE / HORIZONS loader
# ─────────────────────────────────────────────────────────────────────────────

HORIZONS_URL = (
    "https://ssd.jpl.nasa.gov/api/horizons.api"
    "?format=json"
    "&COMMAND='-189'"        # MSL/Curiosity NAIF ID
    "&OBJ_DATA=NO"
    "&MAKE_EPHEM=YES"
    "&EPHEM_TYPE=VECTORS"
    "&CENTER='500@499'"      # Mars center
    "&START_TIME='2024-01-01'"
    "&STOP_TIME='2024-01-02'"
    "&STEP_SIZE='1h'"
    "&OUT_UNITS='KM-S'"
)


def load_spice_horizons(n_frames: int = 24) -> List[TelemetryFrame]:
    """
    Load spacecraft state vectors from NASA HORIZONS API.
    Falls back to circular Mars orbit synthetic if network unavailable.
    """
    try:
        logger.info("Fetching SPICE/HORIZONS vectors from NASA JPL...")
        with urllib.request.urlopen(HORIZONS_URL, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        frames = _parse_horizons(data, n_frames)
        if not frames:
            logger.warning("HORIZONS: 0 state vectors parsed — using synthetic orbital mechanics")
            return _synthetic_orbit(n_frames)
        logger.info(f"HORIZONS: loaded {len(frames)} real state vectors")
        return frames
    except Exception as e:
        logger.warning(f"HORIZONS fetch failed ({e}), using synthetic orbital mechanics")
        return _synthetic_orbit(n_frames)


def _parse_horizons(data: dict, n_frames: int) -> List[TelemetryFrame]:
    frames = []
    result_text = data.get("result", "")
    lines = result_text.splitlines()
    in_data = False
    count = 0
    for line in lines:
        if "$$SOE" in line:
            in_data = True
            continue
        if "$$EOE" in line:
            break
        if not in_data or count >= n_frames:
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            t_jd = float(parts[0])
            x_km, y_km, z_km = float(parts[2]), float(parts[3]), float(parts[4])
            vx, vy, vz       = float(parts[5]), float(parts[6]), float(parts[7])
            frames.append(TelemetryFrame(
                source="SPICE_HORIZONS",
                mission_time_s=t_jd * 86400.0,
                utc_timestamp=parts[1] if len(parts) > 1 else "",
                position_m=np.array([x_km, y_km, z_km]) * 1000.0,
                velocity_ms=np.array([vx, vy, vz]) * 1000.0,
                orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                angular_velocity_rads=np.array([0.0, 0.0, 0.0]),
                temperature_k=240.0,
                thermal_gradient_k_per_s=0.0,
                battery_level=0.8,
                power_consumption_w=400.0,
                solar_irradiance_w_m2=590.0,
                ambient_pressure_pa=0.0,
                wind_speed_ms=0.0,
                uv_index=0.0,
                confidence=0.99,
                sensor_flags={"nav": True},
                metadata={"real_data": True, "jd": t_jd}
            ))
            count += 1
        except (ValueError, IndexError):
            continue
    return frames


def _synthetic_orbit(n_frames: int) -> List[TelemetryFrame]:
    """Circular Mars orbit at 400 km altitude, 2-hour period."""
    R_MARS   = 3_396_200.0   # m
    ALT      = 400_000.0     # m
    r        = R_MARS + ALT
    # v = sqrt(GM/r), GM_Mars = 4.282e13
    v_orbit  = math.sqrt(4.282e13 / r)
    period_s = 2 * math.pi * r / v_orbit
    frames   = []
    dt       = period_s / n_frames

    for i in range(n_frames):
        t    = i * dt
        ang  = 2 * math.pi * t / period_s
        cos_a, sin_a = math.cos(ang), math.sin(ang)

        # Eclipse: battery drops when behind Mars (~40% of orbit)
        in_eclipse = (ang > math.pi * 0.8) and (ang < math.pi * 1.8)
        solar = 0.0 if in_eclipse else 1360.0
        battery = max(0.20, 0.90 - 0.25 * (1.0 if in_eclipse else 0.0))

        # Thermal: cold in eclipse, warm in sunlight
        temp_k = 270.0 if in_eclipse else 310.0

        frames.append(TelemetryFrame(
            source="SYNTHETIC_ORBIT",
            mission_time_s=t,
            utc_timestamp="",
            position_m=np.array([r * cos_a, r * sin_a, 0.0]),
            velocity_ms=np.array([-v_orbit * sin_a, v_orbit * cos_a, 0.0]),
            orientation_quat=np.array([
                math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)]),
            angular_velocity_rads=np.array([0.0, 0.0, 2 * math.pi / period_s]),
            temperature_k=temp_k,
            thermal_gradient_k_per_s=(-1.0 if in_eclipse else 0.5),
            battery_level=battery,
            power_consumption_w=420.0,
            solar_irradiance_w_m2=solar,
            ambient_pressure_pa=0.0,
            wind_speed_ms=0.0,
            uv_index=0.0,
            confidence=0.98,
            sensor_flags={"nav": True, "power": True, "thermal": True},
            metadata={"real_data": False, "model": "circular_orbit",
                      "altitude_km": ALT / 1000, "in_eclipse": in_eclipse}
        ))
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# 3. ESA Rosetta / Mars Express housekeeping
# ─────────────────────────────────────────────────────────────────────────────

ESA_PSA_BASE = (
    "https://archives.esac.esa.int/psa/ftp/MARS-EXPRESS/ASPERA-3/"
    "MEX-M-ASPERA3-2-EDR-ELS-V1.0/DATA/"
)

ESA_FALLBACK_PARAMS = {
    # Mars Express periapsis ~300 km, apoapsis ~10,000 km
    "periapsis_m":  300_000.0 + 3_396_200.0,
    "apoapsis_m":  10_000_000.0 + 3_396_200.0,
    "orbital_period_s": 6.72 * 3600,
    "power_nominal_w": 460.0,
    "temp_nominal_k": 293.0,
}


def load_esa_housekeeping(n_frames: int = 150) -> List[TelemetryFrame]:
    """
    Load ESA Rosetta/Mars Express housekeeping CSV from PSA archive.
    Falls back to elliptical orbit derived from published MEX parameters.
    """
    try:
        logger.info("Fetching ESA PSA housekeeping data...")
        url = ESA_PSA_BASE
        with urllib.request.urlopen(url, timeout=10) as resp:
            raw = resp.read().decode("latin-1")
        frames = _parse_esa_csv(raw, n_frames)
        if not frames:
            logger.warning("ESA PSA: 0 frames parsed — using synthetic MEX elliptical orbit")
            return _synthetic_mex_orbit(n_frames)
        logger.info(f"ESA PSA: loaded {len(frames)} housekeeping frames")
        return frames
    except Exception as e:
        logger.warning(f"ESA PSA fetch failed ({e}), using synthetic MEX elliptical orbit")
        return _synthetic_mex_orbit(n_frames)


def _parse_esa_csv(raw: str, n_frames: int) -> List[TelemetryFrame]:
    frames = []
    reader = csv.DictReader(io.StringIO(raw))
    for i, row in enumerate(reader):
        if i >= n_frames:
            break
        try:
            t    = float(row.get("TIME", i))
            temp = float(row.get("TEMP", 293.0))
            pwr  = float(row.get("POWER", 460.0))
            bat  = float(row.get("BATTERY", 0.8))
            frames.append(TelemetryFrame(
                source="ESA_MEX",
                mission_time_s=t,
                utc_timestamp=row.get("UTC", ""),
                position_m=np.array([0.0, 0.0, 0.0]),
                velocity_ms=np.array([0.0, 0.0, 0.0]),
                orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                angular_velocity_rads=np.array([0.0, 0.0, 0.0]),
                temperature_k=temp,
                thermal_gradient_k_per_s=0.0,
                battery_level=bat,
                power_consumption_w=pwr,
                solar_irradiance_w_m2=590.0,
                ambient_pressure_pa=0.0,
                wind_speed_ms=0.0,
                uv_index=0.0,
                confidence=0.97,
                sensor_flags={"hk": True},
                metadata={"real_data": True}
            ))
        except (ValueError, KeyError):
            continue
    return frames


def _synthetic_mex_orbit(n_frames: int) -> List[TelemetryFrame]:
    """Elliptical Mars Express orbit (300 x 10000 km) derived from published elements."""
    p = ESA_FALLBACK_PARAMS
    r_p, r_a = p["periapsis_m"], p["apoapsis_m"]
    a = (r_p + r_a) / 2          # semi-major axis
    e = (r_a - r_p) / (r_p + r_a)  # eccentricity
    T = p["orbital_period_s"]
    GM = 4.282e13

    frames = []
    dt = T / n_frames

    for i in range(n_frames):
        t = i * dt
        # Mean anomaly → eccentric anomaly (Newton iteration)
        M = 2 * math.pi * t / T
        E = M
        for _ in range(10):
            E = M + e * math.sin(E)
        nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2),
                            math.sqrt(1 - e) * math.cos(E / 2))
        r = a * (1 - e * math.cos(E))

        # Eclipse at apoapsis region (simplified)
        in_eclipse = abs(nu) > math.pi * 0.7
        solar = 0.0 if in_eclipse else 1360.0 / (r / 1.52e11) ** 2

        # Thermal: near periapsis, aerobraking heating
        at_periapsis = r < (r_p * 1.05)
        temp_k = p["temp_nominal_k"] + (40.0 if at_periapsis else 0.0) - \
                 (20.0 if in_eclipse else 0.0)

        battery = max(0.15, 0.85 - 0.30 * (1.0 if in_eclipse else 0.0))

        frames.append(TelemetryFrame(
            source="SYNTHETIC_MEX",
            mission_time_s=t,
            utc_timestamp="",
            position_m=np.array([r * math.cos(nu), r * math.sin(nu), 0.0]),
            velocity_ms=np.array([
                -math.sqrt(GM / (a * (1 - e**2))) * math.sin(nu),
                 math.sqrt(GM / (a * (1 - e**2))) * (e + math.cos(nu)),
                0.0]),
            orientation_quat=np.array([
                math.cos(nu / 2), 0.0, 0.0, math.sin(nu / 2)]),
            angular_velocity_rads=np.array([0.0, 0.0, 2 * math.pi / T]),
            temperature_k=temp_k,
            thermal_gradient_k_per_s=(1.5 if at_periapsis else (-0.3 if in_eclipse else 0.1)),
            battery_level=battery,
            power_consumption_w=p["power_nominal_w"] + (80.0 if at_periapsis else 0.0),
            solar_irradiance_w_m2=solar,
            ambient_pressure_pa=0.0,
            wind_speed_ms=0.0,
            uv_index=0.0,
            confidence=0.96,
            sensor_flags={"nav": True, "power": True, "thermal": True},
            metadata={"real_data": False, "model": "MEX_elliptical",
                      "true_anomaly_deg": math.degrees(nu),
                      "altitude_km": (r - 3_396_200.0) / 1000,
                      "in_eclipse": in_eclipse}
        ))
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# 4. ISS OSDR telemetry
# ─────────────────────────────────────────────────────────────────────────────

ISS_OSDR_URL = (
    "https://osdr.nasa.gov/bio/repo/data/studies/OSD-665/"
    "metadata/OSD-665_metadata_ISS.csv"
)

ISS_FALLBACK_PARAMS = {
    # ISS orbital parameters (published)
    "altitude_m":   408_000.0,
    "inclination_deg": 51.6,
    "orbital_period_s": 5556.0,   # ~92.6 min
    "power_nominal_w": 84_000.0,  # total array
    "crew_module_temp_k": 294.0,
    "battery_dod_max": 0.35,      # max depth of discharge
}


def load_iss_telemetry(n_frames: int = 180) -> List[TelemetryFrame]:
    """
    Load ISS OSDR telemetry CSV.
    Falls back to ISS orbital mechanics from published parameters.
    """
    try:
        logger.info("Fetching ISS OSDR telemetry...")
        req = urllib.request.Request(ISS_OSDR_URL,
                                     headers={"User-Agent": "ARVS-Sim/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
        frames = _parse_iss_csv(raw, n_frames)
        logger.info(f"ISS OSDR: loaded {len(frames)} frames")
        return frames
    except Exception as e:
        logger.warning(f"ISS OSDR fetch failed ({e}), using synthetic ISS orbital data")
        return _synthetic_iss(n_frames)


def _parse_iss_csv(raw: str, n_frames: int) -> List[TelemetryFrame]:
    frames = []
    reader = csv.DictReader(io.StringIO(raw))
    for i, row in enumerate(reader):
        if i >= n_frames:
            break
        try:
            t = float(row.get("ELAPSED_TIME_S", i * 31.0))
            temp = float(row.get("CABIN_TEMP_K", 294.0))
            pwr  = float(row.get("POWER_W", 75000.0))
            bat  = float(row.get("BATTERY_SOC", 0.85))
            frames.append(TelemetryFrame(
                source="ISS_OSDR",
                mission_time_s=t,
                utc_timestamp=row.get("UTC", ""),
                position_m=np.array([0.0, 0.0, 0.0]),
                velocity_ms=np.array([7660.0, 0.0, 0.0]),
                orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                angular_velocity_rads=np.array([0.0, 0.0, 0.0]),
                temperature_k=temp,
                thermal_gradient_k_per_s=0.0,
                battery_level=bat,
                power_consumption_w=pwr,
                solar_irradiance_w_m2=1360.0,
                ambient_pressure_pa=0.0,
                wind_speed_ms=0.0,
                uv_index=0.0,
                confidence=0.99,
                sensor_flags={"hk": True},
                metadata={"real_data": True}
            ))
        except (ValueError, KeyError):
            continue
    return frames


def _synthetic_iss(n_frames: int) -> List[TelemetryFrame]:
    """ISS orbit: 408 km circular, 51.6° inclination, 16 orbits/day."""
    p = ISS_FALLBACK_PARAMS
    R_EARTH   = 6_371_000.0
    r         = R_EARTH + p["altitude_m"]
    T         = p["orbital_period_s"]
    v_orbit   = 2 * math.pi * r / T
    inc_rad   = math.radians(p["inclination_deg"])

    frames = []
    dt = T / n_frames

    for i in range(n_frames):
        t    = i * dt
        ang  = 2 * math.pi * t / T

        cos_a, sin_a = math.cos(ang), math.sin(ang)

        # ISS eclipses ~35% of orbit
        in_eclipse = (ang > math.pi * 1.1) and (ang < math.pi * 1.8)
        solar = 0.0 if in_eclipse else 1360.0

        # Battery: DoD cycles, stays above 65% SoC (35% DoD max)
        soc = p["battery_dod_max"] * (0.5 * (1 + math.cos(ang))) + \
              (1.0 - p["battery_dod_max"])

        # Thermal cycling: -120°C to +120°C external; cabin stays ~21°C
        external_temp_k = 153.0 if in_eclipse else 393.0
        cabin_temp_k    = 294.0 + 0.5 * math.sin(ang)  # tight control

        frames.append(TelemetryFrame(
            source="SYNTHETIC_ISS",
            mission_time_s=t,
            utc_timestamp="",
            position_m=np.array([
                r * cos_a,
                r * sin_a * math.cos(inc_rad),
                r * sin_a * math.sin(inc_rad)]),
            velocity_ms=np.array([
                -v_orbit * sin_a,
                 v_orbit * cos_a * math.cos(inc_rad),
                 v_orbit * cos_a * math.sin(inc_rad)]),
            orientation_quat=np.array([
                math.cos(ang / 2), 0.0, math.sin(ang / 2), 0.0]),
            angular_velocity_rads=np.array([0.0, 0.0, 2 * math.pi / T]),
            temperature_k=cabin_temp_k,
            thermal_gradient_k_per_s=(1.5 if not in_eclipse else -1.5),
            battery_level=soc,
            power_consumption_w=p["power_nominal_w"] * 0.85,
            solar_irradiance_w_m2=solar,
            ambient_pressure_pa=101_325.0,    # pressurised cabin
            wind_speed_ms=0.0,
            uv_index=0.0,
            confidence=0.995,
            sensor_flags={"nav": True, "power": True, "thermal": True,
                          "life_support": True},
            metadata={"real_data": False, "model": "ISS_circular",
                      "altitude_km": p["altitude_m"] / 1000,
                      "in_eclipse": in_eclipse,
                      "external_temp_k": external_temp_k}
        ))
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Master loader — returns all four datasets merged into mission timeline
# ─────────────────────────────────────────────────────────────────────────────

def load_all_telemetry(n_frames_each: int = 200) -> Dict[str, List[TelemetryFrame]]:
    """Load all four telemetry sources. Returns dict keyed by source name."""
    return {
        "REMS":          load_rems(n_frames_each),
        "SPICE_ORBIT":   load_spice_horizons(n_frames_each // 8),
        "ESA_MEX":       load_esa_housekeeping(n_frames_each),
        "ISS":           load_iss_telemetry(n_frames_each),
    }


def frames_to_dataframe(frames: List[TelemetryFrame]) -> pd.DataFrame:
    """Convert a list of TelemetryFrames to a flat DataFrame for analysis."""
    rows = []
    for f in frames:
        rows.append({
            "source":            f.source,
            "mission_time_s":    f.mission_time_s,
            "pos_x":             f.position_m[0],
            "pos_y":             f.position_m[1],
            "pos_z":             f.position_m[2],
            "vel_x":             f.velocity_ms[0],
            "vel_y":             f.velocity_ms[1],
            "vel_z":             f.velocity_ms[2],
            "temperature_k":     f.temperature_k,
            "battery_level":     f.battery_level,
            "power_w":           f.power_consumption_w,
            "solar_w_m2":        f.solar_irradiance_w_m2,
            "pressure_pa":       f.ambient_pressure_pa,
            "wind_ms":           f.wind_speed_ms,
            "confidence":        f.confidence,
            "real_data":         f.metadata.get("real_data", False),
        })
    return pd.DataFrame(rows)
