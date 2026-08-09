"""
Time-Resolved External Laser Pulse Field Engine for Photochemical Pump Dynamics.
"""

import logging
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

def simulate_pump_pulse(
    field_amplitude: float,
    frequency: float,
    fwhm: float,
    t_grid: np.ndarray,
    t0: Optional[float] = None,
    dipoles: Optional[np.ndarray] = None,
    phase: float = 0.0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Computes time-dependent semi-classical laser pulse E(t) and dipole interaction matrices H_int(t).
    
    Args:
        field_amplitude: Peak electric field amplitude E_0 (V/m or a.u.).
        frequency: Pulse central frequency omega (rad/s or Hartree).
        fwhm: Full-width at half-maximum in time (s or a.u.).
        t_grid: 1D array of time points t.
        t0: Pulse center time. Defaults to center of t_grid.
        dipoles: Optional transition dipole matrix of shape (n_states, n_states, 3).
        phase: Carrier-envelope phase phi.
        
    Returns:
        Tuple of (E_field_array, H_int_array)
        - E_field_array: 1D array of shape (len(t_grid),) representing electric field E(t).
        - H_int_array: 3D array of shape (len(t_grid), n_states, n_states) if dipoles provided, else None.
    """
    t_grid = np.asarray(t_grid, dtype=float)
    if t0 is None:
        t0 = (t_grid[0] + t_grid[-1]) / 2.0
        
    # Gaussian temporal envelope standard deviation: sigma = FWHM / (2 * sqrt(2 * ln(2)))
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    
    gaussian_envelope = np.exp(-((t_grid - t0)**2) / (2.0 * max(sigma**2, 1e-18)))
    carrier = np.cos(frequency * (t_grid - t0) + phase)
    
    E_field = field_amplitude * gaussian_envelope * carrier
    
    H_int = None
    if dipoles is not None:
        dipoles = np.asarray(dipoles, dtype=float)
        n_states = dipoles.shape[0]
        n_times = len(t_grid)
        H_int = np.zeros((n_times, n_states, n_states))
        
        # Interaction Hamiltonian H_int(t) = - mu . E(t)
        # Assuming laser polarization along Z-axis (dipoles[:, :, 2]) by default
        mu_z = dipoles[:, :, 2] if dipoles.ndim == 3 else dipoles
        for i in range(n_times):
            H_int[i] = -mu_z * E_field[i]
            
    logger.info("Simulated pump pulse: E0=%.3e, omega=%.3e, FWHM=%.3e", field_amplitude, frequency, fwhm)
    return E_field, H_int
