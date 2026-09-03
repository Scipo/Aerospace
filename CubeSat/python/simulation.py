
import numpy as np
import json
import time as time_module
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from core.kepler import solve_kepler
from core.orbits import (
    OrbitalElements, propagate_orbit, eci_to_ground_track,
    R_EARTH, MU_EARTH, MU_SUN, AU_KM
)
from core.planets import (
    get_planet_elements, PLANET_ELEMENTS_J2000, PLANET_VISUALS
)
from cubesat.subsystems import (
    CubeSatConfig, GroundStation, DEFAULT_GROUND_STATIONS,
    compute_sun_direction, is_in_eclipse, compute_solar_power,
    compute_power_consumption, update_battery,
    compute_ground_station_visibility, compute_attitude_matrix
)


def run_simulation(
    duration_hours: float = 24.0,
    time_step_seconds: float = 30.0,
    cubesat_config: CubeSatConfig = None,
    ground_stations: list = None,
    verbose: bool = True
) -> dict:

    if cubesat_config is None:
        cubesat_config = CubeSatConfig()
    if ground_stations is None:
        ground_stations = DEFAULT_GROUND_STATIONS
    
    # ---- Time setup ----
    duration_s = duration_hours * 3600
    times = np.arange(0, duration_s, time_step_seconds)
    n_steps = len(times)
    
    if verbose:
        print(f"╔══════════════════════════════════════════════╗")
        print(f"║     CubeSat Mission Simulator v1.0          ║")
        print(f"╠══════════════════════════════════════════════╣")
        print(f"║  Duration: {duration_hours:.0f} hours ({n_steps} steps)            ")
        print(f"║  Time step: {time_step_seconds:.0f}s                            ")
        print(f"║  CubeSat: {cubesat_config.name}                ")
        print(f"║  Ground stations: {len(ground_stations)}                      ")
        print(f"╚══════════════════════════════════════════════╝")
    
    t_start = time_module.time()
    
    # ---- Define CubeSat orbit (ISS-like) ----
    cubesat_orbit = OrbitalElements(
        a=R_EARTH + 420,        # 420 km altitude
        e=0.0002,               # Nearly circular
        i=np.radians(51.6),     # ISS inclination
        raan=np.radians(45.0),  # Arbitrary RAAN
        argp=np.radians(0),
        M0=np.radians(0),
        mu=MU_EARTH
    )
    
    if verbose:
        print(f"\nCubeSat orbit:")
        print(f"  Altitude: {cubesat_orbit.a - R_EARTH:.0f} km")
        print(f"  Period: {cubesat_orbit.period/60:.1f} min")
        print(f"  Inclination: {np.degrees(cubesat_orbit.i):.1f}°")
    
    # Step 1: Propagate CubeSat orbit
    if verbose:
        print("\n[1/5] Propagating CubeSat orbit...")
    
    cubesat_positions = propagate_orbit(cubesat_orbit, times)
    ground_track = eci_to_ground_track(cubesat_positions, times)
    
    # Step 2: Propagate planet orbits
    if verbose:
        print("[2/5] Computing planet positions...")

    # Compute at a coarser resolution and interpolate
    planet_times = np.linspace(0, duration_s, min(360, n_steps))
    
    planets_data = {}
    for planet_name in PLANET_ELEMENTS_J2000:
        elem = get_planet_elements(planet_name, t_centuries_from_j2000=0.2639)
        positions_au = propagate_orbit(elem, planet_times)
        
        planets_data[planet_name] = {
            "positions": positions_au.tolist(),
            "times": planet_times.tolist(),
            "orbital_elements": {
                "a_au": elem.a,
                "e": elem.e,
                "i_deg": np.degrees(elem.i),
                "raan_deg": np.degrees(elem.raan),
                "argp_deg": np.degrees(elem.argp),
                "period_days": elem.period / 86400,
            },
            "color": PLANET_VISUALS[planet_name]["color"],
            "radius_km": PLANET_VISUALS[planet_name]["radius_km"],
        }
    
    # Compute full orbit paths for visualization (one complete orbit per planet)
    orbit_paths = {}
    for planet_name in PLANET_ELEMENTS_J2000:
        elem = get_planet_elements(planet_name, t_centuries_from_j2000=0.2639)
        orbit_times = np.linspace(0, elem.period, 360)
        orbit_positions = propagate_orbit(elem, orbit_times)
        orbit_paths[planet_name] = orbit_positions.tolist()
    
    # Step 3: CubeSat subsystem simulation
    if verbose:
        print("[3/5] Simulating CubeSat subsystems...")
    
    # Pre-allocate arrays
    eclipse_flags = np.zeros(n_steps, dtype=bool)
    solar_power = np.zeros(n_steps)
    power_consumption = np.zeros(n_steps)
    battery_level = np.zeros(n_steps)
    
    # Ground station visibility - one flag per station
    gs_visibility = {gs.name: np.zeros(n_steps, dtype=bool) for gs in ground_stations}
    gs_elevation = {gs.name: np.zeros(n_steps) for gs in ground_stations}
    
    # Attitude quaternions (stored as rotation matrices, exported as quaternions)
    attitude_data = np.zeros((n_steps, 4))  # quaternion [w, x, y, z]
    
    # Initialize battery
    battery_wh = cubesat_config.battery_capacity_wh * cubesat_config.battery_initial_soc
    
    # Track statistics
    total_eclipse_time = 0
    total_comms_time = 0
    min_battery = battery_wh
    
    for idx in range(n_steps):
        t = times[idx]
        sat_pos = cubesat_positions[idx]
        
        # Sun direction
        sun_dir = compute_sun_direction(t)
        
        # Eclipse
        in_eclipse = is_in_eclipse(sat_pos, sun_dir)
        eclipse_flags[idx] = in_eclipse
        if in_eclipse:
            total_eclipse_time += time_step_seconds
        
        # Attitude
        att_matrix = compute_attitude_matrix(sat_pos, sun_dir, cubesat_config.pointing_mode)
        # Convert rotation matrix to quaternion for export
        w = 0.5 * np.sqrt(max(0, 1 + att_matrix[0,0] + att_matrix[1,1] + att_matrix[2,2]))
        if w > 1e-6:
            x = (att_matrix[2,1] - att_matrix[1,2]) / (4*w)
            y = (att_matrix[0,2] - att_matrix[2,0]) / (4*w)
            z = (att_matrix[1,0] - att_matrix[0,1]) / (4*w)
        else:
            x, y, z, w = 0, 0, 0, 1
        attitude_data[idx] = [w, x, y, z]
        
        # Power generation
        solar_power[idx] = compute_solar_power(sat_pos, sun_dir, cubesat_config, att_matrix)
        
        # Ground station checks
        any_gs_visible = False
        for gs in ground_stations:
            vis, el = compute_ground_station_visibility(sat_pos, t, gs)
            gs_visibility[gs.name][idx] = vis
            gs_elevation[gs.name][idx] = el
            if vis:
                any_gs_visible = True
                total_comms_time += time_step_seconds / len(ground_stations)
        
        # Power consumption (transmit when ground station visible)
        is_transmitting = any_gs_visible
        payload_active = not in_eclipse and (idx % 10 < 5)  # Payload active 50% in sunlight
        power_consumption[idx] = compute_power_consumption(
            cubesat_config, is_transmitting, payload_active
        )
        
        # Battery update
        battery_wh = update_battery(
            battery_wh, solar_power[idx], power_consumption[idx],
            time_step_seconds, cubesat_config
        )
        battery_level[idx] = battery_wh
        min_battery = min(min_battery, battery_wh)
    
    # Step 4: Compute CubeSat orbit path (full ellipse for visualization)
    if verbose:
        print("[4/5] Computing orbit visualization paths...")
    
    orbit_period = cubesat_orbit.period
    orbit_path_times = np.linspace(0, orbit_period, 360)
    orbit_path_positions = propagate_orbit(cubesat_orbit, orbit_path_times)
    
    # Step 5: Package everything and exporting
    if verbose:
        print("[5/5] Packaging data for export...")
    
    # Downsample positions for JSON
    stride = max(1, n_steps // 2880)
    
    output = {
        "metadata": {
            "simulation_name": f"{cubesat_config.name} Mission Simulation",
            "duration_hours": duration_hours,
            "time_step_seconds": time_step_seconds,
            "total_steps": n_steps,
            "export_stride": stride,
            "exported_steps": len(times[::stride]),
        },
        
        "constants": {
            "R_earth_km": R_EARTH,
            "AU_km": AU_KM,
            "mu_earth": MU_EARTH,
        },
        
        "times": times[::stride].tolist(),
        
        "solar_system": {
            "planets": planets_data,
            "orbit_paths": orbit_paths,
            "sun": {
                "radius_km": 696340,
                "color": "#FDB813",
            }
        },
        
        "cubesat": {
            "config": {
                "name": cubesat_config.name,
                "size_u": cubesat_config.size_u,
                "mass_kg": cubesat_config.mass_kg,
                "pointing_mode": cubesat_config.pointing_mode,
            },
            "orbit": {
                "altitude_km": cubesat_orbit.a - R_EARTH,
                "semi_major_axis_km": cubesat_orbit.a,
                "eccentricity": cubesat_orbit.e,
                "inclination_deg": np.degrees(cubesat_orbit.i),
                "raan_deg": np.degrees(cubesat_orbit.raan),
                "period_minutes": cubesat_orbit.period / 60,
            },
            "orbit_path": orbit_path_positions.tolist(),
            "positions_eci_km": cubesat_positions[::stride].tolist(),
            "ground_track_deg": ground_track[::stride].tolist(),
            "attitude_quaternions": attitude_data[::stride].tolist(),
            
            "subsystems": {
                "eclipse": eclipse_flags[::stride].tolist(),
                "solar_power_w": np.round(solar_power[::stride], 2).tolist(),
                "power_consumption_w": np.round(power_consumption[::stride], 2).tolist(),
                "battery_wh": np.round(battery_level[::stride], 3).tolist(),
                "battery_percent": np.round(
                    battery_level[::stride] / cubesat_config.battery_capacity_wh * 100, 1
                ).tolist(),
            },
            
            "comms": {
                "ground_stations": [
                    {
                        "name": gs.name,
                        "lat_deg": gs.lat_deg,
                        "lon_deg": gs.lon_deg,
                        "visibility": gs_visibility[gs.name][::stride].tolist(),
                    }
                    for gs in ground_stations
                ],
            },
        },
        
        "statistics": {
            "orbits_completed": round(duration_s / cubesat_orbit.period, 1),
            "eclipse_fraction": round(total_eclipse_time / duration_s, 3),
            "eclipse_per_orbit_min": round(
                total_eclipse_time / (duration_s / cubesat_orbit.period) / 60, 1
            ),
            "avg_solar_power_w": round(float(np.mean(solar_power)), 2),
            "avg_power_consumption_w": round(float(np.mean(power_consumption)), 2),
            "min_battery_wh": round(min_battery, 2),
            "min_battery_percent": round(min_battery / cubesat_config.battery_capacity_wh * 100, 1),
            "total_comms_minutes": round(total_comms_time / 60, 1),
            "power_positive": min_battery > 0,
        }
    }
    
    elapsed = time_module.time() - t_start
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"  Simulation complete in {elapsed:.2f}s")
        print(f"{'='*50}")
        print(f"\n  Mission Statistics:")
        print(f"  ─────────────────")
        stats = output["statistics"]
        print(f"  Orbits completed:     {stats['orbits_completed']}")
        print(f"  Eclipse fraction:     {stats['eclipse_fraction']*100:.1f}%")
        print(f"  Eclipse per orbit:    {stats['eclipse_per_orbit_min']:.1f} min")
        print(f"  Avg solar power:      {stats['avg_solar_power_w']:.1f} W")
        print(f"  Avg power draw:       {stats['avg_power_consumption_w']:.1f} W")
        print(f"  Min battery:          {stats['min_battery_wh']:.1f} Wh ({stats['min_battery_percent']:.0f}%)")
        print(f"  Total comms time:     {stats['total_comms_minutes']:.0f} min")
        power_status = "✓ POSITIVE" if stats['power_positive'] else "✗ NEGATIVE"
        print(f"  Power budget:         {power_status}")
        print()
    
    return output


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def export_json(data: dict, filepath: str):
    """Export simulation data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, separators=(',', ':'), cls=NumpyEncoder)
    
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Exported to: {filepath}")
    print(f"  File size: {size_mb:.2f} MB")


# ============================================================
# Main entry point
# ============================================================
if __name__ == "__main__":
    # Run 24-hour simulation
    data = run_simulation(
        duration_hours=24,
        time_step_seconds=30,
        verbose=True
    )
    
    # Export to output folder
    os.makedirs("output", exist_ok=True)
    export_json(data, "output/simulation_data.json")
    
    print("\n  Ready for Three.js visualization!")
