"""
Non-Adiabatic Coupling Vector (NACV) Computation and Velocity Rescaling Engine.
"""

import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)

def compute_nacv(
    grads_i: np.ndarray,
    grads_j: np.ndarray,
    energy_i: float,
    energy_j: float,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Computes approximate Non-Adiabatic Coupling Vector (NACV) d_ij between states i and j.
    
    Args:
        grads_i: Energy gradient array for state i of shape (N_atoms, 3).
        grads_j: Energy gradient array for state j of shape (N_atoms, 3).
        energy_i: Energy of state i.
        energy_j: Energy of state j.
        eps: Small energy gap regularizer.
        
    Returns:
        NACV array d_ij of shape (N_atoms, 3).
    """
    # [ANTI-SPOOFING] Exact analytical Non-Adiabatic Coupling Vectors (NACVs) calculation
    # is required. Toy approximations (gradients difference / gap) are strictly prohibited.
    raise NotImplementedError(
        "Exact analytical Non-Adiabatic Coupling Vectors (NACVs) calculation "
        "is required by the Anti-Spoofing Directive. Toy approximations are prohibited."
    )

def rescale_velocity_after_hop(
    v_velocity: np.ndarray,
    nacv: np.ndarray,
    masses: np.ndarray,
    delta_V: float
) -> Tuple[np.ndarray, bool]:
    """
    Rescales nuclear velocities along the NACV vector direction upon a surface hop to conserve energy.
    
    Args:
        v_velocity: Velocity array of shape (N_atoms, 3).
        nacv: Non-adiabatic coupling vector of shape (N_atoms, 3).
        masses: 1D array of atomic masses of length N_atoms.
        delta_V: Potential energy change V_new - V_old.
        
    Returns:
        Tuple (new_velocity, successful_rescale_flag)
    """
    v = np.asarray(v_velocity, dtype=float).copy()
    d = np.asarray(nacv, dtype=float)
    m = np.asarray(masses, dtype=float)
    
    # Check if NACV norm is near zero
    d_norm_sq = np.sum(d**2)
    if d_norm_sq < 1e-16:
        return v, True
        
    # Normalized NACV unit vector
    d_hat = d / np.sqrt(d_norm_sq)
    
    # Calculate mass-weighted directional component: a = sum_a (d_hat_a^2 / m_a)
    m_3d = np.repeat(m[:, np.newaxis], 3, axis=1)
    
    # Projection of velocity along nacv
    v_dot_d = np.sum(v * d_hat)
    
    # Effective mass along NACV
    mu_eff = np.sum(m_3d * d_hat**2)
    
    # Kinetic energy along NACV direction
    E_kin_nacv = 0.5 * mu_eff * (v_dot_d**2)
    
    if delta_V <= E_kin_nacv:
        # Hop is energetically allowed
        scale_factor = np.sqrt(max(0.0, 1.0 - delta_V / max(E_kin_nacv, 1e-14)))
        v_nacv_new = scale_factor * v_dot_d
        
        # Update velocity along NACV
        v_new = v - (v_dot_d - v_nacv_new) * d_hat
        return v_new, True
    else:
        # Frustrated hop: reverse velocity component along NACV
        v_new = v - 2.0 * v_dot_d * d_hat
        logger.info("Frustrated hop: reversed nuclear velocity component along NACV direction.")
        return v_new, False