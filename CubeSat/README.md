# CubeSat Mission Simulator

A complete aerospace simulation and 3D visualization project combining orbital mechanics, CubeSat subsystem modeling, and a solar system orrery.

Built with **Python** (orbital engine) and **Three.js** (3D visualization).

---

## Quick Start

```bash
# 1. Install Python dependency
pip install numpy

# 2. Run the simulation (generates data)
cd python
python simulation.py

# 3. Copy data to frontend
cp output/simulation_data.json ../frontend/

# 4. Serve the frontend
cd ../frontend
python -m http.server 8000

# 5. Open browser at http://localhost:8000
```

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.8+ | Orbital mechanics engine |
| NumPy | 1.20+ | Numerical computation |
| Modern browser | Chrome/Firefox/Edge/Safari | Three.js visualization |

```bash
python --version
pip install numpy
```

No other dependencies. Three.js loads from CDN automatically.

---
---

## Controls

| Action | Input |
|--------|-------|
| Rotate view | Click + drag |
| Zoom | Scroll wheel |
| Earth Orbit view | Click EARTH ORBIT button |
| Solar System view | Click SOLAR SYSTEM button |
| Ground Track map | Click GROUND TRACK button |
| Play / Pause | ▶ / ⏸ button |
| Scrub timeline | Drag timeline slider |
| Change speed | + / − buttons (0.25x to 100x) |

---

## Customization

### Change orbit (in python/simulation.py)

```python
cubesat_orbit = OrbitalElements(
    a=R_EARTH + 420,        # Altitude in km
    e=0.0002,               # Eccentricity
    i=np.radians(51.6),     # Inclination
    raan=np.radians(45.0),  # RAAN
    argp=np.radians(0),     # Argument of periapsis
    M0=np.radians(0),       # Starting mean anomaly
    mu=MU_EARTH
)
```

### Change CubeSat config (in python/cubesat/subsystems.py)

```python
cell_efficiency: float = 0.29       # Solar cell efficiency
battery_capacity_wh: float = 20.0   # Battery size
pointing_mode: str = "nadir"        # or "sun_pointing", "tumbling"
```

After changes: re-run simulation, copy JSON, refresh browser.

---

## Troubleshooting

**"Could not load simulation_data.json"**: Ensure you copied the JSON to frontend/ and are using `python -m http.server` (not opening the HTML file directly).

**Blank screen**: Check browser console (F12). Ensure WebGL is supported.

**ModuleNotFoundError**: Run from the python/ directory: `cd python && python simulation.py`

---

## Learning Resources

- [orbital-mechanics.space](https://orbital-mechanics.space/) — Free orbital mechanics textbook
- [orbital-mechanic](https://oer.pressbooks.pub/lynnanegeorge/) - Free orbital mechanics textbook
- [AMSAT Keplerian Tutorial](https://www.amsat.org/keplerian-elements-tutorial/) — Beginner-friendly
- [NASA CubeSat 101 PDF](https://www.nasa.gov/wp-content/uploads/2017/03/nasa_csli_cubesat_101_508.pdf)
- [Three.js Manual](https://threejs.org/manual/)
- [Discover Three.js](https://discoverthreejs.com/) — Free tutorial series
