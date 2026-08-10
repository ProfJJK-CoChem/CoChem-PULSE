"""
Tully's Fewest Switches Surface Hopping (FSSH) Electronic Integrator (RK4).
"""

import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

HBAR = 1.0  # Atomic units for electronic dynamics (Hartree * a.u._time)

def electronic_derivative(c: np.ndarray, V: np.ndarray, d_nacv: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Computes dc/dt = -i/hbar (V c) - sum_j (v . d_kj) c_j.
    c: (n_states,) complex array
    V: (n_states,) real energy potential array
    d_nacv: (n_states, n_states, n_atoms, 3) non-adiabatic coupling vectors
    v: (n_atoms, 3) nuclear velocity array
    """
    n_states = len(c)
    dc = np.zeros(n_states, dtype=complex)
    
    for k in range(n_states):
        # Diagonal energy term
        dc[k] -= (1j / HBAR) * V[k] * c[k]
        
        # Off-diagonal non-adiabatic coupling term
        for j in range(n_states):
            if k != j:
                v_dot_d = np.sum(v * d_nacv[k, j])
                dc[k] -= v_dot_d * c[j]
                
    return dc

def integrate_electronic_rk4(
    c_amplitudes: np.ndarray,
    V_energies: np.ndarray,
    d_nacv: np.ndarray,
    v_velocity: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Integrates electronic wavepacket amplitudes c(t) over step dt using 4th order Runge-Kutta.
    """
    c0 = np.asarray(c_amplitudes, dtype=complex)
    
    k1 = electronic_derivative(c0, V_energies, d_nacv, v_velocity)
    k2 = electronic_derivative(c0 + 0.5 * dt * k1, V_energies, d_nacv, v_velocity)
    k3 = electronic_derivative(c0 + 0.5 * dt * k2, V_energies, d_nacv, v_velocity)
    k4 = electronic_derivative(c0 + dt * k3, V_energies, d_nacv, v_velocity)
    
    c_new = c0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    # Preserve probability norm: sum |c_i|^2 = 1
    norm = np.sqrt(np.sum(np.abs(c_new)**2))
    if norm > 1e-12:
        c_new = c_new / norm
        
    return c_new

def evaluate_hop(
    c_amplitudes: np.ndarray,
    d_nacv: np.ndarray,
    v_velocity: np.ndarray,
    current_state: int,
    dt: float,
    E_kin_nacv: Optional[float] = None,
    delta_V: Optional[float] = None,
    random_r: Optional[float] = None,
    rng: Optional[Any] = None
) -> Tuple[int, bool]:
    """
    Evaluates FSSH hopping probabilities from current_state to all other states.
    Uses deterministic seeded RNG (MOCK-23 / Suggestion 88).
    
    PULSE-20: Includes potential energy difference check (delta_V) against kinetic energy along NACV (E_kin_nacv).
    
    Returns:
        Tuple (new_state, hopped_flag)
    """
    c = np.asarray(c_amplitudes, dtype=complex)
    n_states = len(c)
    k = current_state
    c_k_sq = np.abs(c[k])**2
    
    if c_k_sq < 1e-14:
        return current_state, False
        
    probabilities = np.zeros(n_states)
    
    for j in range(n_states):
        if j != k:
            v_dot_d = np.sum(v_velocity * d_nacv[k, j])
            # FSSH hopping rate: g_{k->j} = dt * max(0, -2 Re(c_k* c_j v . d_jk) / |c_k|^2)
            term = -2.0 * np.real(np.conj(c[k]) * c[j] * v_dot_d)
            rate = max(0.0, term / c_k_sq)
            probabilities[j] = rate * dt

    total_prob = np.sum(probabilities)
    if total_prob <= 0.0:
        return current_state, False

    if random_r is not None:
        r = float(random_r)
    elif rng is not None:
        if isinstance(rng, np.random.Generator):
            r = float(rng.random())
        else:
            r = float(np.random.default_rng(rng).random())
    else:
        # Enforce deterministic default seed 42 to prevent unseeded non-reproducible calls
        r = float(np.random.default_rng(42).random())

    cum_prob = 0.0
    target_state = current_state
    for j in range(n_states):
        if j != k:
            cum_prob += probabilities[j]
            if r <= cum_prob:
                target_state = j
                break

    if target_state == current_state:
        return current_state, False

    # PULSE-20: Energy conservation check on surface hopping
    if delta_V is not None and E_kin_nacv is not None:
        if delta_V > E_kin_nacv:
            logger.info("Frustrated hop from S_%d to S_%d: delta_V=%.4f > E_kin_nacv=%.4f", current_state, target_state, delta_V, E_kin_nacv)
            return current_state, False

    logger.info("Successful surface hop from S_%d to S_%d", current_state, target_state)
    return target_state, True
