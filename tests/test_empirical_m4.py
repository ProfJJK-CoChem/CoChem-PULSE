"""
Empirical Stress Test Suite for Milestone 4 (CoChem-KINETIC & CoChem-PULSE).
Validates:
1. Landau-Zener probability formula at zero coupling (v12=0 => P_diab=1.0).
2. VTST bottleneck search and Troe fall-off pressure dependencies.
3. FSSH surface hopping trajectory propagation for bit-for-bit reproducibility when passing identical seeds.
4. Velocity Verlet MD trajectory integration for energy conservation (delta E / E < 10^-4) over 500 steps.
"""

import sys
import pytest
import numpy as np

from kinetic_core.thermo import (
    landau_zener_probability,
    calculate_vtst_rate,
    troe_falloff_rate
)
from pulse_core.dispatcher import PulseDispatcher
from pulse_core.aimnet2_nse import AIMNet2NSECalculator


def test_landau_zener_zero_coupling_empirical():
    """1. Test Landau-Zener probability formula at zero coupling (v12=0 => P_diab=1.0)."""
    p_diab_zero = landau_zener_probability(v12=0.0, force_diff=1.0, velocity=1e5, return_type="diabatic")
    p_adiab_zero = landau_zener_probability(v12=0.0, force_diff=1.0, velocity=1e5, return_type="adiabatic")
    assert p_diab_zero == 1.0, f"Expected P_diab=1.0 at v12=0, got {p_diab_zero}"
    assert p_adiab_zero == 0.0, f"Expected P_adiab=0.0 at v12=0, got {p_adiab_zero}"

    # Small v12 near zero
    p_diab_near_zero = landau_zener_probability(v12=1e-15, force_diff=1.0, velocity=1e5, return_type="diabatic")
    assert p_diab_near_zero == 1.0, f"Expected P_diab=1.0 for small v12, got {p_diab_near_zero}"

    # Finite coupling
    p_diab_finite = landau_zener_probability(v12=0.1, force_diff=2.0, velocity=1000.0, return_type="diabatic")
    p_adiab_finite = landau_zener_probability(v12=0.1, force_diff=2.0, velocity=1000.0, return_type="adiabatic")
    assert np.isclose(p_diab_finite + p_adiab_finite, 1.0), "P_diab + P_adiab should sum to 1.0"
    assert 0.0 <= p_diab_finite < 1.0, "P_diab should be in [0, 1)"

    # Zero velocity
    p_zero_vel = landau_zener_probability(v12=0.1, force_diff=2.0, velocity=0.0, return_type="diabatic")
    assert p_zero_vel == 1.0, f"Expected P_diab=1.0 for velocity=0, got {p_zero_vel}"


def test_vtst_and_troe_dependencies_empirical():
    """2. Test VTST bottleneck search and Troe fall-off pressure dependencies."""
    # 2A. VTST Bottleneck Search
    s_coords = np.linspace(-1.0, 1.0, 21)
    energies = 10.0 - 5.0 * (s_coords - 0.4)**2
    res_vtst = calculate_vtst_rate(
        irc_s_coords=s_coords,
        irc_energies=energies,
        imaginary_freq=400.0,
        temp=300.0
    )
    assert res_vtst['bottleneck_index'] == 14, f"Expected bottleneck index 14, got {res_vtst['bottleneck_index']}"
    assert np.isclose(res_vtst['s_star_amu_ang'], 0.4), f"Expected s*=0.4, got {res_vtst['s_star_amu_ang']}"
    assert res_vtst['k_vtst'] > 0.0, "Rate constant must be positive"

    # 2B. Troe Fall-Off Pressure Dependencies
    k_0 = 1.0     # Low-pressure limit rate coefficient (atm^-1 s^-1)
    k_inf = 100.0  # High-pressure limit rate constant (s^-1)
    
    pressures = np.logspace(-4, 6, 20)  # 10^-4 to 10^6 atm
    k_rates = [troe_falloff_rate(k_0, k_inf, P, temp=298.15, F_cent=0.6) for P in pressures]
    
    # Low pressure limit scaling check
    k_1e5 = troe_falloff_rate(k_0, k_inf, P_atm=1e-5, temp=298.15)
    k_1e6 = troe_falloff_rate(k_0, k_inf, P_atm=1e-6, temp=298.15)
    ratio_low = k_1e5 / k_1e6
    assert np.isclose(ratio_low, 10.0, rtol=1e-2), f"Expected low-pressure rate ratio to be ~10.0, got {ratio_low}"
    
    # High pressure limit check: k(P) -> k_inf
    k_high = troe_falloff_rate(k_0, k_inf, P_atm=1e6, temp=298.15)
    assert np.isclose(k_high, k_inf, rtol=1e-2), f"At high pressure, k should approach k_inf ({k_inf})"

    # Monotonicity check
    is_monotonic = all(k_rates[i] <= k_rates[i+1] for i in range(len(k_rates)-1))
    assert is_monotonic, "Troe fall-off rate must increase monotonically with pressure"


def test_fssh_bit_for_bit_reproducibility_empirical():
    """3. Test FSSH surface hopping trajectory propagation for bit-for-bit reproducibility when passing identical seeds."""
    ref_coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.2]
    ])
    symbols = ["C", "C"]
    
    dispatcher = PulseDispatcher()
    
    # Run 1 with seed=42
    res1 = dispatcher.run_pulse_pipeline(
        ref_coords=ref_coords,
        atomic_symbols=symbols,
        n_trajectories=5,
        n_steps=20,
        dt_fs=0.5,
        seed=42
    )
    
    # Run 2 with seed=42
    res2 = dispatcher.run_pulse_pipeline(
        ref_coords=ref_coords,
        atomic_symbols=symbols,
        n_trajectories=5,
        n_steps=20,
        dt_fs=0.5,
        seed=42
    )
    
    # Run 3 with seed=99
    res3 = dispatcher.run_pulse_pipeline(
        ref_coords=ref_coords,
        atomic_symbols=symbols,
        n_trajectories=5,
        n_steps=20,
        dt_fs=0.5,
        seed=99
    )

    assert np.array_equal(res1["coordinates"], res2["coordinates"]), "Coordinates must be bit-for-bit identical"
    assert np.array_equal(res1["velocities"], res2["velocities"]), "Velocities must be bit-for-bit identical"
    assert np.array_equal(res1["populations"], res2["populations"]), "Populations must be bit-for-bit identical"

    assert not np.array_equal(res1["coordinates"], res3["coordinates"]), "Different seeds must produce different sampling"


def test_velocity_verlet_energy_conservation_empirical():
    """4. Test Velocity Verlet MD trajectory integration for energy conservation (delta E / E < 10^-4) over 500 steps."""
    calc = AIMNet2NSECalculator(n_states=3)
    ref_coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.2]
    ])
    n_atoms = len(ref_coords)
    atomic_masses = np.full(n_atoms, 12.0)
    masses_au = atomic_masses * 1836.15
    
    dt_fs = 0.1  # small time step 0.1 fs
    dt_au = dt_fs * 41.341374575751  # fs to atomic time units
    n_steps = 500
    
    curr_coords = ref_coords.copy()
    curr_vels = np.zeros((n_atoms, 3))
    curr_vels[0, 2] = 0.001  # Bohr / a.u. time
    curr_vels[1, 2] = -0.001
    
    BOHR_TO_ANGSTROM = 0.529177210903
    curr_state = 0
    energies = []
    
    for step in range(n_steps):
        V_energies, gradients, _ = calc.evaluate_surfaces(curr_coords)
        e_pot = V_energies[curr_state]  # Hartree
        e_kin = 0.5 * np.sum(masses_au[:, np.newaxis] * curr_vels**2)  # Hartree
        e_tot = e_pot + e_kin
        energies.append(e_tot)
        
        # Velocity Verlet step
        grad_curr = gradients[curr_state]
        acc_n = -grad_curr / masses_au[:, np.newaxis]
        
        vel_half = curr_vels + 0.5 * acc_n * dt_au
        curr_coords = curr_coords + vel_half * dt_au * BOHR_TO_ANGSTROM
        
        V_next, grad_next, _ = calc.evaluate_surfaces(curr_coords)
        acc_next = -grad_next[curr_state] / masses_au[:, np.newaxis]
        curr_vels = vel_half + 0.5 * acc_next * dt_au

    energies = np.array(energies)
    e_0 = energies[0]
    max_delta_e = np.max(np.abs(energies - e_0))
    rel_error = max_delta_e / abs(e_0)
    
    assert rel_error < 1e-4, f"Energy conservation failed! Relative error {rel_error:.6e} >= 1e-4"
