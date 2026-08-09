"""
Wigner Phase-Space Initial Condition Sampling Engine for Quantum Dynamics.
"""

import logging
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Constants in Atomic Units / Physical Constants
HBAR = 1.054571817e-34  # J s
KB = 1.380649e-23        # J / K
AMU_TO_KG = 1.66053906660e-27  # kg
ANGSTROM_TO_M = 1e-10
CM1_TO_RAD_S = 2.99792458e10 * 2.0 * np.pi

def sample_wigner(
    coords: np.ndarray,
    hessian: Optional[np.ndarray] = None,
    n_trajectories: int = 100,
    freqs: Optional[np.ndarray] = None,
    normal_modes: Optional[np.ndarray] = None,
    masses: Optional[np.ndarray] = None,
    temp: float = 300.0,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Wigner quantum phase-space samples (coords and momenta) for initial condition trajectory ensembles.
    
    Args:
        coords: Reference geometry array of shape (N_atoms, 3) in Angstroms.
        hessian: Cartesian mass-weighted Hessian array of shape (3N, 3N) in Hartree/Bohr^2 or J/(kg m^2).
        n_trajectories: Number of trajectory initial conditions to sample.
        freqs: Optional 1D array of vibrational frequencies in cm^-1.
        normal_modes: Optional normal mode transformation matrix of shape (3N, 3N).
        masses: 1D array of atomic masses in amu (length N_atoms). Defaults to carbon (12.0) if None.
        temp: Temperature in Kelvin (default 300.0).
        random_seed: Optional seed for reproducible sampling.
        
    Returns:
        Tuple (sampled_coords, sampled_momenta) each of shape (n_trajectories, N_atoms, 3).
    """
    rng = np.random.default_rng(random_seed)
        
    coords = np.asarray(coords, dtype=float)
    n_atoms = coords.shape[0]
    n_dof = 3 * n_atoms
    
    if masses is None:
        masses = np.full(n_atoms, 12.0)
    else:
        masses = np.asarray(masses, dtype=float)
        
    # Construct mass vector for 3N DOF
    m_3n = np.repeat(masses, 3)  # amu
    m_3n_kg = m_3n * AMU_TO_KG
    
    if freqs is None or normal_modes is None:
        if hessian is None:
            # Build an acoustic harmonic approximation if no Hessian is provided
            logger.warning("No Hessian provided; creating simple harmonic approximation based on coordinates.")
            hessian = np.eye(n_dof) * 0.1
            
        # Mass-weight Hessian: H_mw = M^{-1/2} H M^{-1/2}
        inv_sqrt_m = 1.0 / np.sqrt(m_3n_kg)
        mw_hessian = hessian * np.outer(inv_sqrt_m, inv_sqrt_m)
        
        evals, normal_modes = np.linalg.eigh(mw_hessian)
        # Filter negative / small eigenvalues (rotations/translations)
        evals_pos = np.maximum(evals, 1e-8)
        omega = np.sqrt(evals_pos)  # rad/s approx
    else:
        omega = freqs * CM1_TO_RAD_S
        normal_modes = np.asarray(normal_modes, dtype=float)
        
    sampled_coords = np.zeros((n_trajectories, n_atoms, 3))
    sampled_momenta = np.zeros((n_trajectories, n_atoms, 3))
    
    beta = 1.0 / (KB * max(temp, 1.0))
    
    for i in range(n_trajectories):
        q_mode = np.zeros(n_dof)
        p_mode = np.zeros(n_dof)
        
        for k in range(n_dof):
            w_k = max(omega[k], 1e-4)
            # Quantum Wigner variances: sigma_q^2 = (hbar / (2 w)) coth(beta hbar w / 2)
            x_k = HBAR * w_k * beta / 2.0
            coth_k = 1.0 / np.tanh(min(x_k, 50.0))
            
            sigma_q = np.sqrt((HBAR / (2.0 * w_k)) * coth_k)  # mass-weighted coordinate variance
            sigma_p = np.sqrt((HBAR * w_k / 2.0) * coth_k)     # mass-weighted momentum variance
            
            # Exact Quantum Harmonic Oscillator Wigner probability distribution sampling via Box-Muller transform
            u1 = float(rng.uniform(1e-12, 1.0))
            u2 = float(rng.uniform(0.0, 1.0))
            mag = np.sqrt(-2.0 * np.log(u1))
            q_mode[k] = mag * np.cos(2.0 * np.pi * u2) * sigma_q
            p_mode[k] = mag * np.sin(2.0 * np.pi * u2) * sigma_p
            
        # Transform back to Cartesian coordinates & momenta
        dq_cart_mw = normal_modes @ q_mode  # mass-weighted displacements
        dp_cart_mw = normal_modes @ p_mode  # mass-weighted momenta
        
        # Un-mass-weight: dq = dq_mw / sqrt(m), dp = dp_mw * sqrt(m)
        dq_cart = (dq_cart_mw * inv_sqrt_m).reshape(n_atoms, 3) / ANGSTROM_TO_M
        dp_cart = (dp_cart_mw / inv_sqrt_m).reshape(n_atoms, 3)  # kg m/s
        
        sampled_coords[i] = coords + dq_cart
        sampled_momenta[i] = dp_cart
        
    logger.info("Sampled %d Wigner initial conditions for %d atoms at T=%.1f K.", n_trajectories, n_atoms, temp)
    return sampled_coords, sampled_momenta
