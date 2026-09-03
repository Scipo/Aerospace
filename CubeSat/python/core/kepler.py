"""
Kepler's Equation Solver
========================
Solves M = E - e*sin(E) for E given M and e
using Newton-Raphson iteration.
"""

import numpy as np


def solve_kepler(M: float, e: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    """
    Solve Kepler's equation: M = E - e * sin(E)
    """
    # Normalize M to [0, 2π]
    M = M % (2 * np.pi)
    
    # Initial guess: E = M is never more than distance e from the solution
    if e < 0.8:
        E = M
    else:
        E = np.pi  # Better starting point for high eccentricity
    
    # Newton-Raphson iteration
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1.0 - e * np.cos(E)
        
        delta = f / f_prime
        E -= delta
        
        if abs(delta) < tol:
            return E
    
    # Should rarely reach here for e < 1
    return E


def solve_kepler_vectorized(M_array: np.ndarray, e: float, tol: float = 1e-12) -> np.ndarray:
    """
    Solve Kepler's equation for an array of mean anomalies.
    """
    return np.array([solve_kepler(M, e, tol) for M in M_array])


def eccentric_to_true_anomaly(E: float, e: float) -> float:
    """
    Convert eccentric anomaly E to true anomaly ν.
    """
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(1.0 - e) * np.cos(E / 2.0)
    )
    return nu % (2 * np.pi)


def orbital_radius(a: float, e: float, E: float) -> float:
    """
    Compute the distance r from the central body.
    """
    return a * (1.0 - e * np.cos(E))


# Testing
if __name__ == "__main__":
    # Test: circular orbit (e=0), E should equal M
    for M_test in [0, 0.5, 1.0, np.pi, 5.0]:
        E_test = solve_kepler(M_test, 0.0)
        assert abs(E_test - (M_test % (2*np.pi))) < 1e-10, f"Failed for circular orbit M={M_test}"
    
    # Test: verify solution satisfies Kepler's equation
    import random
    random.seed(42)
    for _ in range(10000):
        e_test = random.random() * 0.95
        M_test = random.random() * 2 * np.pi
        E_test = solve_kepler(M_test, e_test)
        residual = E_test - e_test * np.sin(E_test) - M_test
        assert abs(residual) < 1e-10, f"Residual too large: {residual}"
    
    print("✓ All Kepler solver tests passed (10,000 random cases)")
    print("Everything pass")
