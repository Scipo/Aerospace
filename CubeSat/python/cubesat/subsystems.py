"""
CubeSat Subsystem Models
========================
Models for power generation, battery, eclipse detection,
ground station visibility, and attitude modes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from core.orbits import R_EARTH


# ============================================================
# CubeSat Configuration
# ============================================================

@dataclass
class CubeSatConfig:
    """Physical and electrical parameters of a CubeSat"""
    
    # Form factor
    name: str = "3U"
    size_u: int = 3  # 1U, 2U, 3U, 6U
    mass_kg: float = 4.0
    
    # Dimensions (meters) - for a 3U
    length: float = 0.3   # 30 cm
    width: float = 0.1    # 10 cm  
    height: float = 0.1   # 10 cm
    
    # Solar panels
    cell_efficiency: float = 0.29       # 29% multijunction
    panel_area_per_face_m2: float = 0.01  # 100 cm² per 1U face
    num_solar_faces: int = 5            # 5 faces can have cells (1 has connector)
    has_deployable_panels: bool = True
    deployable_panel_area_m2: float = 0.06  # total deployable area
    
    # Battery
    battery_capacity_wh: float = 20.0   # Watt-hours
    battery_initial_soc: float = 0.9    # Start at 90% charge
    charge_efficiency: float = 0.9      # Charging losses
    
    # Subsystem power consumption (Watts)
    power_cdh: float = 0.5         # Command & data handling (always on)
    power_adcs_active: float = 1.5  # Attitude control when active
    power_adcs_idle: float = 0.1    # ADCS idle
    power_comms_tx: float = 4.0     # Transmitter during ground pass
    power_comms_rx: float = 0.5     # Receiver (always listening)
    power_payload: float = 3.0      # Payload (camera/sensor) when active
    
    # Attitude
    pointing_mode: str = "nadir"    # "tumbling", "sun_pointing", "nadir"


@dataclass 
class GroundStation:
    """Ground station for communication"""
    name: str
    lat_deg: float
    lon_deg: float
    min_elevation_deg: float = 5.0  # Minimum elevation for contact
    
    @property
    def position_ecef(self) -> np.ndarray:
        """Ground station position in ECEF (km)"""
        lat = np.radians(self.lat_deg)
        lon = np.radians(self.lon_deg)
        return R_EARTH * np.array([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat)
        ])


# Default ground stations
DEFAULT_GROUND_STATIONS = [
    GroundStation("Houston", 29.7604, -95.3698),
    GroundStation("Goldstone", 35.4267, -116.8900),
    GroundStation("Canberra", -35.4017, 148.9817),
    GroundStation("Madrid", 40.4312, -3.9528),
    GroundStation("Sofia", 42.6977, 23.3219),
]


# ============================================================
# Eclipse Detection
# ============================================================

def compute_sun_direction(t_seconds: float, sim_start_centuries: float = 0.2639) -> np.ndarray:
    """
    Compute Sun direction vector as seen from Earth in ECI frame.
    """

    # Sun's mean longitude and mean anomaly (simplified)
    T = sim_start_centuries + t_seconds / (86400 * 365.25 * 100)
    
    # Mean longitude of Sun (degrees)
    L0 = 280.46646 + 36000.76983 * T
    # Mean anomaly (degrees)
    M = 357.52911 + 35999.05029 * T
    M_rad = np.radians(M % 360)
    
    # Equation of center (approximate)
    C = (1.914602 - 0.004817 * T) * np.sin(M_rad) + \
        0.019993 * np.sin(2 * M_rad)
    
    # Sun's ecliptic longitude
    sun_lon = np.radians((L0 + C) % 360)
    
    # Obliquity of ecliptic
    obliquity = np.radians(23.439291 - 0.0130042 * T)
    
    # Sun direction in ECI (unit vector FROM Earth TO Sun)
    sun_dir = np.array([
        np.cos(sun_lon),
        np.sin(sun_lon) * np.cos(obliquity),
        np.sin(sun_lon) * np.sin(obliquity)
    ])
    
    return sun_dir / np.linalg.norm(sun_dir)


def is_in_eclipse(sat_pos_eci: np.ndarray, sun_dir: np.ndarray) -> bool:
    """
    Determine if satellite is in Earth's shadow.
    True if in eclipse
    """

    # Project satellite position onto sun direction
    proj = np.dot(sat_pos_eci, sun_dir)
    
    # If satellite is on the Sun-side of Earth, not in eclipse
    if proj > 0:
        return False
    
    # Perpendicular distance from Earth-Sun line
    perp = sat_pos_eci - proj * sun_dir
    perp_dist = np.linalg.norm(perp)
    
    # In shadow if perpendicular distance < Earth's radius
    return perp_dist < R_EARTH


# ============================================================
# Power Model
# ============================================================

def compute_solar_power(sat_pos_eci: np.ndarray, sun_dir: np.ndarray,
                        config: CubeSatConfig, 
                        attitude_matrix: np.ndarray = None) -> float:
    """
    Compute instantaneous solar power generation.
    Returns power_watts
    """
    # Check eclipse first
    if is_in_eclipse(sat_pos_eci, sun_dir):
        return 0.0
    
    SOLAR_IRRADIANCE = 1361.0  # W/m² at 1 AU
    
    if attitude_matrix is not None:
        # Compute power for each face based on actual orientation
        face_normals_body = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], dtype=float)
        
        total_power = 0.0
        for fn in face_normals_body:
            # Transform face normal to ECI
            fn_eci = attitude_matrix @ fn
            # Cosine of angle between face normal and sun direction
            cos_theta = np.dot(fn_eci, sun_dir)
            if cos_theta > 0:  # Face is illuminated
                area = config.panel_area_per_face_m2
                total_power += config.cell_efficiency * area * SOLAR_IRRADIANCE * cos_theta
        
        # Add deployable panels if present
        if config.has_deployable_panels:
            total_power += (config.cell_efficiency * 
                          config.deployable_panel_area_m2 * 
                          SOLAR_IRRADIANCE * 0.7)
        
        return total_power
    else:
        total_area = (config.num_solar_faces * config.panel_area_per_face_m2)
        if config.has_deployable_panels:
            total_area += config.deployable_panel_area_m2
        
        effective_area = total_area * 0.35
        return config.cell_efficiency * effective_area * SOLAR_IRRADIANCE


def compute_power_consumption(config: CubeSatConfig, 
                               is_transmitting: bool = False,
                               payload_active: bool = False,
                               adcs_active: bool = True) -> float:
    """Compute total power consumption in Watts"""
    power = config.power_cdh + config.power_comms_rx
    power += config.power_adcs_active if adcs_active else config.power_adcs_idle
    if is_transmitting:
        power += config.power_comms_tx
    if payload_active:
        power += config.power_payload
    return power


def update_battery(current_wh: float, power_gen: float, power_use: float,
                   dt_seconds: float, config: CubeSatConfig) -> float:
    """
    Update battery state of charge.
    Returns new battery level in Wh.
    """
    dt_hours = dt_seconds / 3600.0
    
    net_power = power_gen * config.charge_efficiency - power_use
    new_wh = current_wh + net_power * dt_hours
    
    # Clamp to [0, capacity]
    return np.clip(new_wh, 0.0, config.battery_capacity_wh)


# ============================================================
# Ground Station Visibility
# ============================================================

EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s

def compute_ground_station_visibility(sat_pos_eci: np.ndarray, 
                                       t_seconds: float,
                                       station: GroundStation,
                                       gmst0: float = 0.0) -> Tuple[bool, float]:
    """
    Check if satellite is visible from a ground station.
    """
    # Rotate ground station position from ECEF to ECI
    gmst = gmst0 + EARTH_ROTATION_RATE * t_seconds
    cos_g = np.cos(gmst)
    sin_g = np.sin(gmst)
    
    gs_ecef = station.position_ecef
    gs_eci = np.array([
        gs_ecef[0] * cos_g - gs_ecef[1] * sin_g,
        gs_ecef[0] * sin_g + gs_ecef[1] * cos_g,
        gs_ecef[2]
    ])
    
    # Vector from ground station to satellite
    los = sat_pos_eci - gs_eci
    los_dist = np.linalg.norm(los)
    
    # Local up direction at ground station
    up = gs_eci / np.linalg.norm(gs_eci)
    
    # Elevation angle
    sin_el = np.dot(los, up) / los_dist
    elevation = np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))
    
    visible = elevation >= station.min_elevation_deg
    return visible, elevation


# ============================================================
# Attitude (simplified)
# ============================================================

def compute_attitude_matrix(sat_pos_eci: np.ndarray, 
                             sun_dir: np.ndarray,
                             mode: str = "nadir") -> np.ndarray:
    """
    Compute the attitude rotation matrix (body frame → ECI).
    
    Modes:
    - "nadir": -Z body axis points at Earth center
    - "sun_pointing": +Z body axis points at Sun
    - "tumbling": random orientation
    """
    if mode == "nadir":
        # -Z points toward Earth (nadir)
        z_body = -sat_pos_eci / np.linalg.norm(sat_pos_eci)

        # Use sun direction as reference to break ambiguity
        x_body = np.cross(z_body, sun_dir)
        x_norm = np.linalg.norm(x_body)
        if x_norm < 1e-10:
            x_body = np.cross(z_body, np.array([0, 0, 1]))
            x_norm = np.linalg.norm(x_body)
        x_body /= x_norm
        
        # Y completes the right-hand system
        y_body = np.cross(z_body, x_body)
        
        return np.column_stack([x_body, y_body, z_body])
    
    elif mode == "sun_pointing":
        # +Z points at Sun
        z_body = sun_dir.copy()
        
        # X perpendicular
        ref = np.array([0, 0, 1])
        if abs(np.dot(z_body, ref)) > 0.99:
            ref = np.array([1, 0, 0])
        x_body = np.cross(z_body, ref)
        x_body /= np.linalg.norm(x_body)
        y_body = np.cross(z_body, x_body)
        
        return np.column_stack([x_body, y_body, z_body])
    
    elif mode == "tumbling":
        # Slow tumble - rotation based on time encoded in position
        angle = np.linalg.norm(sat_pos_eci) * 0.01  # pseudo-random from position
        c, s = np.cos(angle), np.sin(angle)
        # Simple rotation around a diagonal axis
        R = np.array([
            [c, -s, 0],
            [s*0.7, c*0.7, -0.714],
            [s*0.714, 0.714*c, c*0.7]
        ])
        # Orthogonalize via QR
        Q, _ = np.linalg.qr(R)
        return Q
    
    # Default: identity (aligned with ECI)
    return np.eye(3)


# Testing
if __name__ == "__main__":
    config = CubeSatConfig()
    print(f"CubeSat: {config.name}")
    print(f"  Size: {config.size_u}U")
    print(f"  Mass: {config.mass_kg} kg")
    print(f"  Battery: {config.battery_capacity_wh} Wh")
    
    # Test eclipse detection
    sun_dir = np.array([1, 0, 0])  # Sun along +x
    
    # Satellite on sun side - not in eclipse
    sat_sunside = np.array([7000, 1000, 0])
    assert not is_in_eclipse(sat_sunside, sun_dir)
    
    # Satellite behind Earth - in eclipse
    sat_shadow = np.array([-7000, 0, 0])
    assert is_in_eclipse(sat_shadow, sun_dir)
    
    # Satellite behind Earth but offset - not in eclipse
    sat_offset = np.array([-7000, 8000, 0])
    assert not is_in_eclipse(sat_offset, sun_dir)
    
    print("  Eclipse detection: ✓")
    
    # Test power generation
    p_sun = compute_solar_power(sat_sunside, sun_dir, config)
    p_shadow = compute_solar_power(sat_shadow, sun_dir, config)
    print(f"  Power in sunlight: {p_sun:.1f} W")
    print(f"  Power in eclipse: {p_shadow:.1f} W")
    assert p_sun > 0
    assert p_shadow == 0
    
    # Test power consumption
    p_idle = compute_power_consumption(config, is_transmitting=False, payload_active=False)
    p_tx = compute_power_consumption(config, is_transmitting=True, payload_active=True)
    print(f"  Power consumption (idle): {p_idle:.1f} W")
    print(f"  Power consumption (TX+payload): {p_tx:.1f} W")
    
    # Test ground station visibility  
    gs = GroundStation("Test", 0, 0)
    sat_overhead = np.array([R_EARTH + 400, 0, 0])
    vis, el = compute_ground_station_visibility(sat_overhead, 0, gs)
    print(f"  Ground station visibility (overhead): {vis}, elevation: {el:.1f}°")
    assert vis
    
    print("\n✓ All CubeSat subsystem tests passed")
    print("\n Everything Pass")

