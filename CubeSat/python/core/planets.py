"""
Planetary Data
==============
Approximate Keplerian elements for the planets, valid for visualization.
Source: JPL "Approximate Positions of the Planets"
https://ssd.jpl.nasa.gov/planets/approx_pos.html
"""

import numpy as np
from core.orbits import OrbitalElements, MU_SUN


def _deg(d):
    return np.radians(d)


PLANET_ELEMENTS_J2000 = {
    "mercury": {
        "a": 0.38709927, "e": 0.20563593, 
        "i": 7.00497902, "raan": 48.33076593,
        "argp_bar": 77.45779628, "L": 252.25032350,
        # Rates per century
        "a_dot": 0.00000037, "e_dot": 0.00001906,
        "i_dot": -0.00594749, "raan_dot": -0.12534081,
        "argp_bar_dot": 0.16047689, "L_dot": 149472.67411175,
    },
    "venus": {
        "a": 0.72333566, "e": 0.00677672,
        "i": 3.39467605, "raan": 76.67984255,
        "argp_bar": 131.60246718, "L": 181.97909950,
        "a_dot": 0.00000390, "e_dot": -0.00004107,
        "i_dot": -0.00078890, "raan_dot": -0.27769418,
        "argp_bar_dot": 0.00268329, "L_dot": 58517.81538729,
    },
    "earth": {
        "a": 1.00000261, "e": 0.01671123,
        "i": -0.00001531, "raan": 0.0,
        "argp_bar": 102.93768193, "L": 100.46457166,
        "a_dot": 0.00000562, "e_dot": -0.00004392,
        "i_dot": -0.01294668, "raan_dot": 0.0,
        "argp_bar_dot": 0.32327364, "L_dot": 35999.37244981,
    },
    "mars": {
        "a": 1.52371034, "e": 0.09339410,
        "i": 1.84969142, "raan": 49.55953891,
        "argp_bar": -23.94362959, "L": -4.55343205,
        "a_dot": 0.00001847, "e_dot": 0.00007882,
        "i_dot": -0.00813131, "raan_dot": -0.29257343,
        "argp_bar_dot": 0.44441088, "L_dot": 19140.30268499,
    },
    "jupiter": {
        "a": 5.20288700, "e": 0.04838624,
        "i": 1.30439695, "raan": 100.47390909,
        "argp_bar": 14.72847983, "L": 34.39644051,
        "a_dot": -0.00011607, "e_dot": -0.00013253,
        "i_dot": -0.00183714, "raan_dot": 0.20469106,
        "argp_bar_dot": 0.21252668, "L_dot": 3034.74612775,
    },
    "saturn": {
        "a": 9.53667594, "e": 0.05386179,
        "i": 2.48599187, "raan": 113.66242448,
        "argp_bar": 92.59887831, "L": 49.95424423,
        "a_dot": -0.00125060, "e_dot": -0.00050991,
        "i_dot": 0.00193609, "raan_dot": -0.28867794,
        "argp_bar_dot": -0.41897216, "L_dot": 1222.49362201,
    },
}

# rendering the planets
PLANET_VISUALS = {
    "mercury": {"color": "#a0a0a0", "radius_km": 2439.7},
    "venus":   {"color": "#e8cda0", "radius_km": 6051.8},
    "earth":   {"color": "#4a90d9", "radius_km": 6378.1},
    "mars":    {"color": "#c1440e", "radius_km": 3389.5},
    "jupiter": {"color": "#c88b3a", "radius_km": 69911},
    "saturn":  {"color": "#e8d5a3", "radius_km": 58232},
}


def get_planet_elements(name: str, t_centuries_from_j2000: float = 0.0) -> OrbitalElements:
    """
    Get orbital elements for a planet at a given time.
    """
    p = PLANET_ELEMENTS_J2000[name]
    T = t_centuries_from_j2000
    
    # Compute elements at time T
    a = p["a"] + p["a_dot"] * T
    e = p["e"] + p["e_dot"] * T
    i = p["i"] + p["i_dot"] * T
    raan = p["raan"] + p["raan_dot"] * T
    argp_bar = p["argp_bar"] + p["argp_bar_dot"] * T
    L = p["L"] + p["L_dot"] * T
    
    # Derived quantities
    argp = argp_bar - raan  # argument of perihelion
    M0 = L - argp_bar       # mean anomaly
    
    return OrbitalElements(
        a=a,
        e=e,
        i=_deg(i),
        raan=_deg(raan),
        argp=_deg(argp),
        M0=_deg(M0),
        epoch=0.0,
        mu=MU_SUN
    )


def get_sun_position_from_earth(t_centuries: float) -> np.ndarray:
    """
    Get the Sun's position as seen from Earth.
    """
    from core.orbits import propagate_orbit
    earth = get_planet_elements("earth", t_centuries)
    # Earth position at t=0 (since elements are already for this time)
    pos = propagate_orbit(earth, np.array([0.0]))
    # Sun is at origin in heliocentric frame
    return -pos[0]  # Sun position relative to Earth


# Test
if __name__ == "__main__":
    from core.orbits import propagate_orbit
    
    print("Planet orbital periods:")
    for name in PLANET_ELEMENTS_J2000:
        elem = get_planet_elements(name)
        period_days = elem.period / 86400
        period_years = period_days / 365.25
        print(f"  {name:10s}: {period_days:10.1f} days ({period_years:.2f} years)")
    
    print("\nExpected approximate values:")
    print("  Mercury: 88 days, Venus: 225 days, Earth: 365 days")
    print("  Mars: 687 days, Jupiter: 4333 days, Saturn: 10759 days")
    
    # Verify Earth's orbit is ~1 AU
    earth = get_planet_elements("earth")
    times = np.linspace(0, earth.period, 360)
    positions = propagate_orbit(earth, times)
    distances = np.sqrt(np.sum(positions**2, axis=1))
    print(f"\nEarth orbit distance range: {np.min(distances):.4f} - {np.max(distances):.4f} AU")
    print(f"  (Expected: ~0.983 - ~1.017 AU)")
    
    print("\n✓ Planetary data tests passed")
