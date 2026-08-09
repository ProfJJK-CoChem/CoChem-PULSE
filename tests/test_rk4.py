"""
Unit tests for RK4 Electronic Integrator and FSSH Hopping Evaluation.
"""

import pytest
import numpy as np
from pulse_core.fssh_rk4 import integrate_electronic_rk4, evaluate_hop

def test_rk4_norm_preservation():
    c_init = np.array([0.6 + 0.0j, 0.8 + 0.0j], dtype=complex)
    V_energies = np.array([0.0, 0.2])
    d_nacv = np.zeros((2, 2, 2, 3))
    d_nacv[0, 1] = 0.1
    d_nacv[1, 0] = -0.1
    v_vel = np.ones((2, 3))
    dt = 0.1
    
    c_next = integrate_electronic_rk4(c_init, V_energies, d_nacv, v_vel, dt)
    
    norm_sq = np.sum(np.abs(c_next)**2)
    assert np.isclose(norm_sq, 1.0, atol=1e-6)

def test_evaluate_hop_frustrated():
    c_init = np.array([0.707 + 0.0j, 0.707 + 0.0j], dtype=complex)
    d_nacv = np.zeros((2, 2, 2, 3))
    d_nacv[0, 1] = 0.2
    d_nacv[1, 0] = -0.2
    v_vel = np.ones((2, 3))
    dt = 0.5
    
    # Large energy difference delta_V > E_kin_nacv -> frustrated hop
    new_state, hopped = evaluate_hop(
        c_init, d_nacv, v_vel, current_state=0, dt=dt,
        E_kin_nacv=0.01, delta_V=0.5, random_r=0.001
    )
    
    assert new_state == 0
    assert not hopped
