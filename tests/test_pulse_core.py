import hashlib
from typing import Any, Dict, List, Optional
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

def test_swarm_manager_validation() -> None:
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

def test_laser_simulation() -> None:
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

def test_nacv_and_velocity_scaling() -> None:
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

def test_decoherence_and_kill_switch() -> None:
    c_amplitudes = np.array([0.707 + 0.0j, 0.707 + 0.0j], dtype=complex)
    E_potentials = np.array([0.0, 0.1])
    c_damped = apply_granucci_persico_decoherence(c_amplitudes, 0, E_potentials, E_kin=0.05, dt=0.1)
    assert np.isclose(np.sum(np.abs(c_damped)**2), 1.0)
    
    # Dissociation check
    bound_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    dissoc_coords = np.array([[0.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    
    assert not check_dissociation(bound_coords)
    assert check_dissociation(dissoc_coords)

def test_aimnet2_nse_calculator() -> None:
    from pulse_core.aimnet2_nse import AIMNet2NSECalculator, ModelNotLoadedError
    calc = AIMNet2NSECalculator(n_states=3)
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]])
    v_energies, gradients, d_nacv = calc.evaluate_surfaces(coords)
    assert len(v_energies) == 3
    assert gradients.shape == (3, 2, 3)
    assert d_nacv.shape == (3, 3, 2, 3)
    assert issubclass(ModelNotLoadedError, Exception)


def test_viz_and_dispatcher() -> None:
    import h5py
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
        
        with h5py.File(h5_path, "r") as h5:
            assert "pulse/trajectories/state_chaining" in h5
            assert "pulse/trajectories/rotational_constants" in h5


def test_aimnet2_nse_physical_gap_and_provenance() -> None:
    from pulse_core.aimnet2_nse import AIMNet2NSECalculator
    calc = AIMNet2NSECalculator(n_states=3)
    coords1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    coords2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    
    v1, g1, d1 = calc.evaluate_surfaces(coords1)
    v2, g2, d2 = calc.evaluate_surfaces(coords2)
    
    # Energy gap V1 - V0 should depend on geometry (non-constant)
    gap1 = v1[1] - v1[0]
    gap2 = v2[1] - v2[0]
    assert not np.isclose(gap1, gap2)
    assert calc.provenance_tag in ["[E]", "[M]", "[D]"]


def test_jensen_inequality_rotational_constants() -> None:
    from pulse_core.wigner import compute_vibrational_averaged_constants, compute_inertia_tensor
    rng = np.random.default_rng(42)
    frames = rng.normal(0, 0.1, (50, 3, 3)) + np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    masses = np.array([16.0, 1.0, 1.0])
    
    B_e, B_0, meta = compute_vibrational_averaged_constants(frames, masses)
    assert len(B_e) == 3
    assert len(B_0) == 3
    assert meta["averaging_method"] == "INVERSE_TENSOR_AVERAGE_JENSEN"
    assert meta["provenance_tag"] == "[D]"
    assert meta["jensen_satisfied"] is True
    
    # Verify Jensen's inequality <1/I_k> >= 1/<I_k> directly in principal axes
    I_list = [compute_inertia_tensor(f, masses) for f in frames]
    I_eig_list = [np.sort(np.linalg.eigvalsh(I_t)) for I_t in I_list]
    I_avg_eig = np.mean(I_eig_list, axis=0)
    inv_I_avg_eig = 1.0 / np.maximum(I_avg_eig, 1e-10)
    B_inv_I_avg = np.sort(505379.006 * inv_I_avg_eig)[::-1]

    assert np.all(B_0 >= B_inv_I_avg - 1e-6)
    assert not np.allclose(B_0, B_inv_I_avg)


def test_hdf5_state_chaining_and_rotational_constants() -> None:
    import h5py
    from pulse_core.serializer import serialize_swarm_to_hdf5
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, "test_state_chaining.h5")
        swarm_data = {
            "coordinates": np.zeros((2, 5, 2, 3)),
            "velocities": np.zeros((2, 5, 2, 3)),
            "populations": np.ones((2, 5, 3)),
            "provenance_tag": "[D]"
        }
        sc_data = {
            "state_in": "T2-3h_r2SCAN-3c_TightOpt",
            "state_out": "T4-1d_FSSH_Trajectory",
            "inherited_geometry_hash": "abc123hash",
            "inherited_hessian_file": "hessian.json"
        }
        rc_data = {
            "B_e_mhz": np.array([1000.0, 500.0, 333.3]),
            "B_0_mhz": np.array([1005.0, 502.0, 334.0]),
            "averaging_method": "INVERSE_TENSOR_AVERAGE_JENSEN",
            "provenance_tag": "[D]"
        }
        serialize_swarm_to_hdf5(h5_path, swarm_data, state_chaining=sc_data, rotational_constants=rc_data)
        
        with h5py.File(h5_path, "r") as h5:
            sc_grp = h5["pulse/trajectories/state_chaining"]
            assert sc_grp.attrs["state_in"] == "T2-3h_r2SCAN-3c_TightOpt"
            assert sc_grp.attrs["inherited_geometry_hash"] == "abc123hash"
            
            rc_grp = h5["pulse/trajectories/rotational_constants"]
            assert "B_e_mhz" in rc_grp
            assert rc_grp.attrs["averaging_method"] == "INVERSE_TENSOR_AVERAGE_JENSEN"
            assert rc_grp.attrs["provenance_tag"] == "[D]"


def test_cuda_mps_payload_config() -> None:
    coords = np.zeros((2, 3, 3))
    momenta = np.zeros((2, 3, 3))
    jobs = dispatch_swarm_to_node((coords, momenta), engine="AIMNet2-NSE", use_mps=True, thread_percentage=25)
    assert len(jobs) == 2
    assert "[CUDA_MPS]" in jobs[0]["tags"]
    assert jobs[0]["mps_config"]["use_mps"] is True
    assert jobs[0]["mps_config"]["thread_percentage"] == 25


def test_diatomic_linear_rotor_jensen_inequality() -> None:
    """
    Unit Test for 2-atom linear rotors (CO) verifying singular axial inertia tensor handling.
    Ensures safe_invert_inertia_tensor prevents floating point overflow and Loewner ordering mismatch.
    """
    from pulse_core.wigner import compute_vibrational_averaged_constants
    
    co_coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.128]
    ])
    co_masses = np.array([12.000, 15.995])
    
    rng = np.random.default_rng(42)
    displacements = rng.normal(0.0, 0.05, (100, 2, 3))
    frames = co_coords[np.newaxis, :, :] + displacements
    
    B_e, B_0, meta = compute_vibrational_averaged_constants(frames, co_masses)
    
    assert len(B_e) == 3
    assert len(B_0) == 3
    assert meta["averaging_method"] == "INVERSE_TENSOR_AVERAGE_JENSEN"
    assert meta["jensen_satisfied"] is True
    assert B_0[0] >= B_0[1] - 1e-6
    assert B_0[1] >= B_0[2] - 1e-6
    # Axial rotational constant must not overflow to 10^15+ MHz
    assert B_0[0] < 1e10



def calculate_artifact_sha256(filepath: str | Path) -> str:
    """Calculates SHA-256 hash of a computational artifact."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Artifact file not found: {filepath}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()