"""
Integration and Unit Tests for CoChem-PULSE Suite.
"""

import pytest
import os
import tempfile
import numpy as np
from pulse_core.swarm_manager import dispatch_swarm_to_node
from pulse_core.laser import simulate_pump_pulse
from pulse_core.nacv import compute_nacv, rescale_velocity_after_hop
from pulse_core.decoherence import apply_granucci_persico_decoherence
from pulse_core.kill_switch import check_dissociation
from pulse_core.viz import plot_population_dynamics, render_trajectory_3d
from pulse_core.serializer import serialize_swarm_to_hdf5
from pulse_core.dispatcher import PulseDispatcher

def test_swarm_manager_validation():
    coords = np.zeros((10, 3, 3))
    momenta = np.zeros((10, 3, 3))
    
    # Valid dispatch
    jobs = dispatch_swarm_to_node((coords, momenta), engine="AIMNet2-NSE")
    assert len(jobs) == 10
    assert jobs[0]["engine"] == "AIMNet2-NSE"
    
    # Invalid tuple
    with pytest.raises(TypeError):
        dispatch_swarm_to_node(coords)
        
    # Mismatched length
    with pytest.raises(ValueError):
        dispatch_swarm_to_node((coords[:5], momenta))

def test_laser_simulation():
    t_grid = np.linspace(0, 100, 50)
    E_field, H_int = simulate_pump_pulse(
        field_amplitude=0.01,
        frequency=0.1,
        fwhm=20.0,
        t_grid=t_grid,
        dipoles=np.ones((2, 2, 3))
    )
    assert len(E_field) == 50
    assert H_int.shape == (50, 2, 2)

def test_nacv_and_velocity_scaling():
    nacv = compute_nacv(
        grads_i=np.array([[0.1, 0.0, 0.0]]),
        grads_j=np.array([[0.3, 0.0, 0.0]]),
        energy_i=0.0,
        energy_j=0.1
    )
    assert nacv.shape == (1, 3)
    
    v_init = np.array([[1.0, 0.0, 0.0]])
    masses = np.array([12.0])
    v_scaled, success = rescale_velocity_after_hop(v_init, nacv, masses, delta_V=0.01)
    assert success
    assert v_scaled.shape == (1, 3)

def test_decoherence_and_kill_switch():
    c_amplitudes = np.array([0.707 + 0.0j, 0.707 + 0.0j], dtype=complex)
    E_potentials = np.array([0.0, 0.1])
    c_damped = apply_granucci_persico_decoherence(c_amplitudes, 0, E_potentials, E_kin=0.05, dt=0.1)
    assert np.isclose(np.sum(np.abs(c_damped)**2), 1.0)
    
    # Dissociation check
    bound_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    dissoc_coords = np.array([[0.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    
    assert not check_dissociation(bound_coords)
    assert check_dissociation(dissoc_coords)

def test_aimnet2_nse_calculator():
    from pulse_core.aimnet2_nse import AIMNet2NSECalculator
    calc = AIMNet2NSECalculator(n_states=3)
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]])
    v_energies, gradients, d_nacv = calc.evaluate_surfaces(coords)
    assert len(v_energies) == 3
    assert gradients.shape == (3, 2, 3)
    assert d_nacv.shape == (3, 3, 2, 3)

def test_viz_and_dispatcher():
    dispatcher = PulseDispatcher()
    ref_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "test_pulse_state.h5")
        res = dispatcher.run_pulse_pipeline(
            ref_coords=ref_coords,
            atomic_symbols=["C", "H"],
            n_trajectories=2,
            n_steps=5,
            output_h5=h5_path
        )
        assert res["coordinates"].shape == (2, 5, 2, 3)
        assert os.path.exists(h5_path)

