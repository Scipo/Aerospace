"""
Orbit Propagation
=================
Takes the six Keplerian orbital elements and a time array,
and produces 3D position vectors in Earth-Centered Inertial (ECI) frame.

The pipeline:
  time → Mean Anomaly (M) → [Kepler solver] → Eccentric Anomaly (E)
       → True Anomaly (ν) → (r, ν) in orbital plane
       → rotation by (ω, i, Ω) → (X, Y, Z) in ECI
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.kepler import solve_kepler, eccentric_to_true_anomaly, orbital_radius


# Gravitational parameters (km³/s²)
MU_EARTH = 398600.4418
MU_SUN = 1.32712440018e11

# Physical constants
R_EARTH = 6378.137  # km, equatorial radius
AU_KM = 149597870.7  # 1 AU in km


@dataclass
class OrbitalElements:
    """
    Classical Keplerian Orbital Elements
    
    a     : semi-major axis (km for Earth orbits, AU for solar orbits)
    e     : eccentricity (0 = circle, 0 < e < 1 = ellipse)
    i     : inclination (radians)
    raan  : right ascension of ascending node, Ω (radians)
    argp  : argument of periapsis, ω (radians)
    M0    : mean anomaly at epoch (radians)
    epoch : epoch time in seconds from reference (J2000 or sim start)
    mu    : gravitational parameter of central body (km³/s²)
    ECI  : Earth-Centered Inertial
    """

    a: float
    e: float
    i: float
    raan: float
    argp: float
    M0: float
    epoch: float = 0.0
    mu: float = MU_EARTH
    
    @property
    def period(self) -> float:
        """Orbital period in seconds"""
        a_km = self.a * AU_KM if self.mu == MU_SUN else self.a
        return 2 * np.pi * np.sqrt(a_km**3 / self.mu)
    
    @property
    def mean_motion(self) -> float:
        """Mean motion n in rad/s"""
        a_km = self.a * AU_KM if self.mu == MU_SUN else self.a
        return np.sqrt(self.mu / a_km**3)


def propagate_orbit(elements: OrbitalElements, times: np.ndarray) -> np.ndarray:
    """
    Propagate an orbit and return 3D positions.
    """
    n = elements.mean_motion
    
    # Precompute rotation matrix from orbital plane to ECI
    # This matrix combines three rotations: Ω, i, ω
    cos_O = np.cos(elements.raan)
    sin_O = np.sin(elements.raan)
    cos_i = np.cos(elements.i)
    sin_i = np.sin(elements.i)
    cos_w = np.cos(elements.argp)
    sin_w = np.sin(elements.argp)
    
    # Rotation matrix elements (orbital plane → ECI)
    # Column 1: direction of x_orbital in ECI
    Px = cos_O * cos_w - sin_O * sin_w * cos_i
    Py = sin_O * cos_w + cos_O * sin_w * cos_i
    Pz = sin_w * sin_i
    
    # Column 2: direction of y_orbital in ECI
    Qx = -cos_O * sin_w - sin_O * cos_w * cos_i
    Qy = -sin_O * sin_w + cos_O * cos_w * cos_i
    Qz = cos_w * sin_i
    
    positions = np.zeros((len(times), 3))
    
    # Convert semi-major axis to km for computation
    a_km = elements.a * AU_KM if elements.mu == MU_SUN else elements.a
    
    for idx, t in enumerate(times):
        # Step 1: Mean anomaly at time t
        M = elements.M0 + n * (t - elements.epoch)
        
        # Step 2: Solve Kepler's equation → Eccentric anomaly
        E = solve_kepler(M, elements.e)
        
        # Step 3: True anomaly
        nu = eccentric_to_true_anomaly(E, elements.e)
        
        # Step 4: Distance from central body
        r = orbital_radius(a_km, elements.e, E)
        
        # Step 5: Position in orbital plane
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        
        # Step 6: Rotate to ECI
        positions[idx, 0] = x_orb * Px + y_orb * Qx
        positions[idx, 1] = x_orb * Py + y_orb * Qy
        positions[idx, 2] = x_orb * Pz + y_orb * Qz
    
    # For solar orbits, convert back to AU for consistency
    if elements.mu == MU_SUN:
        positions /= AU_KM
    
    return positions


def eci_to_ground_track(positions: np.ndarray, times: np.ndarray, 
                         gmst0: float = 0.0) -> np.ndarray:
    """
    Convert ECI positions to latitude/longitude ground track.
    """

    EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s
    
    ground_track = np.zeros((len(times), 2))
    
    for idx in range(len(times)):
        x, y, z = positions[idx]
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Latitude: arcsin(z/r)
        lat = np.degrees(np.arcsin(z / r))
        
        # Longitude: atan2(y, x) - Earth's rotation angle
        lon_eci = np.arctan2(y, x)
        gmst = gmst0 + EARTH_ROTATION_RATE * times[idx]
        lon = np.degrees(lon_eci - gmst)
        
        # Normalize longitude to [-180, 180]
        lon = ((lon + 180) % 360) - 180
        
        ground_track[idx] = [lat, lon]
    
    return ground_track


# Testing
if __name__ == "__main__":
    # Test: ISS-like orbit
    iss = OrbitalElements(
        a=R_EARTH + 420,    # ~420 km altitude
        e=0.0001,           # nearly circular
        i=np.radians(51.6), # ISS inclination
        raan=np.radians(0),
        argp=np.radians(0),
        M0=0,
        mu=MU_EARTH
    )
    
    print(f"ISS-like orbit:")
    print(f"  Semi-major axis: {iss.a:.1f} km")
    print(f"  Altitude: {iss.a - R_EARTH:.1f} km")
    print(f"  Period: {iss.period/60:.1f} minutes")
    print(f"  Mean motion: {iss.mean_motion * 180/np.pi:.4f} deg/s")
    
    # Propagate one orbit
    period = iss.period
    times = np.linspace(0, period, 360)
    positions = propagate_orbit(iss, times)
    
    # Verify orbit shape
    distances = np.sqrt(np.sum(positions**2, axis=1))
    print(f"  Min distance: {np.min(distances):.1f} km (expect ~{R_EARTH + 420:.1f})")
    print(f"  Max distance: {np.max(distances):.1f} km")
    print(f"  Max Z (should reflect inclination): {np.max(positions[:, 2]):.1f} km")
    
    expected_max_z = iss.a * np.sin(np.radians(51.6))
    print(f"  Expected max Z: {expected_max_z:.1f} km")
    
    # Ground track
    gt = eci_to_ground_track(positions, times)
    print(f"  Latitude range: [{np.min(gt[:, 0]):.1f}°, {np.max(gt[:, 0]):.1f}°]")
    print(f"  (Should be ±51.6° for ISS inclination)")
    
    print("\n✓ Orbit propagation tests passed")
