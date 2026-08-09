"""
Granucci-Persico Energy-Based Decoherence Correction Engine for Surface Hopping.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

HBAR = 1.0  # Atomic units

def apply_granucci_persico_decoherence(
    c_amplitudes: np.ndarray,
    current_state: int,
    E_potentials: np.ndarray,
    E_kin: float,
    dt: float,
    C_param: float = 0.1
) -> np.ndarray:
    """
    Applies Granucci-Persico energy-based decoherence damping to electronic amplitudes c(t).
    
    Args:
        c_amplitudes: 1D complex array of electronic state amplitudes.
        current_state: Index of the current active electronic state.
        E_potentials: 1D real array of potential energies for all states.
        E_kin: Nuclear kinetic energy.
        dt: Time step duration.
        C_param: Empirical decoherence constant (default 0.1 a.u. / Hartree).
        
    Returns:
        Damped and re-normalized electronic amplitude array.
    """
    c = np.asarray(c_amplitudes, dtype=complex).copy()
    n_states = len(c)
    k_curr = current_state
    E_curr = E_potentials[k_curr]
    
    E_kin_pos = max(E_kin, 1e-6)
    
    for j in range(n_states):
        if j != k_curr:
            delta_E = abs(E_potentials[j] - E_curr)
            if delta_E > 1e-12:
                # Decoherence time tau_j = (hbar / delta_E) * (1 + C / E_kin)
                tau_j = (HBAR / delta_E) * (1.0 + C_param / E_kin_pos)
                damping = np.exp(-dt / tau_j)
                c[j] *= damping
                
    # Re-normalize current active state amplitude to conserve total norm = 1
    sum_other_sq = np.sum(np.abs(c[np.arange(n_states) != k_curr])**2)
    if sum_other_sq < 1.0:
        phase = np.angle(c[k_curr]) if np.abs(c[k_curr]) > 1e-14 else 0.0
        c[k_curr] = np.sqrt(1.0 - sum_other_sq) * np.exp(1j * phase)
    else:
        norm = np.sqrt(np.sum(np.abs(c)**2))
        if norm > 1e-14:
            c /= norm
            
    return c
