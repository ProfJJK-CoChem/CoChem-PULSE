"""
Master Dispatcher and Pipeline Coordinator for CoChem-PULSE.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from .swarm_manager import dispatch_swarm_to_node
from .wigner import sample_wigner
from .fssh_rk4 import integrate_electronic_rk4, evaluate_hop
from .nacv import rescale_velocity_after_hop
from .decoherence import apply_granucci_persico_decoherence
from .kill_switch import check_dissociation
from .serializer import serialize_swarm_to_hdf5

from .aimnet2_nse import AIMNet2NSECalculator

logger = logging.getLogger(__name__)

class PulseDispatcher:
    """
    Coordinates ingestion of upstream TOPOS/LUMOS geometries and executes
    Non-Adiabatic Surface Hopping (NASH) trajectory dynamics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.aimnet2_calc = AIMNet2NSECalculator(n_states=3)
        
    def run_pulse_pipeline(
        self,
        ref_coords: np.ndarray,
        atomic_symbols: List[str],
        n_trajectories: int = 20,
        n_steps: int = 50,
        dt_fs: float = 0.5,
        temp_k: float = 300.0,
        output_h5: Optional[str] = None,
        seed: int = 42,
        state_in: str = "T2-3h_r2SCAN-3c_TightOpt",
        state_out: str = "T4-1d_FSSH_Trajectory",
        inherited_geometry_hash: Optional[str] = None,
        inherited_hessian_file: Optional[str] = None,
        use_mps: bool = True,
        thread_percentage: int = 25
    ) -> Dict[str, Any]:
        """
        Executes complete surface hopping trajectory ensemble calculation with CUDA MPS co-scheduling,
        Jensen inverse inertia rotational constants, and 11-arrow state chaining pipeline integration.
        """
        ref_coords = np.asarray(ref_coords, dtype=float)
        n_atoms = len(ref_coords)
        rng = np.random.default_rng(seed)
        
        logger.info("Starting PULSE trajectory swarm pipeline for %d atoms (%d trajectories, %d steps).", n_atoms, n_trajectories, n_steps)
        
        # Add performance warnings
        if thread_percentage < 50:
            logger.warning("Performance Warning: thread_percentage is set below 50%% (%d%%). This may bottleneck the swarm execution.", thread_percentage)
        if not use_mps:
            logger.warning("Performance Warning: CUDA MPS co-scheduling is disabled. GPU utilization may be suboptimal.")
        
        # 1. Wigner Phase-Space Sampling
        coords_sampled, momenta_sampled = sample_wigner(
            coords=ref_coords,
            n_trajectories=n_trajectories,
            temp=temp_k,
            random_seed=seed
        )
        
        # 2. Swarm Job Preparation with CUDA MPS Co-Scheduler Integration (§8A.4 / PULSE-04)
        jobs = dispatch_swarm_to_node(
            wigner_seeds=(coords_sampled, momenta_sampled),
            engine="AIMNet2-NSE",
            use_mps=use_mps,
            thread_percentage=thread_percentage
        )
        
        # 3. Trajectory Propagation Loop (Symplectic Velocity Verlet Integration)
        dt_au = dt_fs * 41.341374575751  # fs to atomic time units
        n_states = 3
        all_coords = np.zeros((n_trajectories, n_steps, n_atoms, 3))
        all_vels = np.zeros((n_trajectories, n_steps, n_atoms, 3))
        all_pops = np.zeros((n_trajectories, n_steps, n_states))
        atomic_masses = np.full(n_atoms, 12.0)  # amu approximation
        masses_au = atomic_masses * 1836.15
        
        for traj_idx in range(n_trajectories):
            curr_coords = coords_sampled[traj_idx].copy()
            curr_vels = momenta_sampled[traj_idx].copy() / 12.0
            
            # Initial state amplitudes: 100% in S1 (excited state)
            c_amplitudes = np.array([0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
            curr_state = 1
            
            for step in range(n_steps):
                all_coords[traj_idx, step] = curr_coords
                all_vels[traj_idx, step] = curr_vels
                all_pops[traj_idx, step] = np.abs(c_amplitudes)**2
                
                # Check kill-switch
                if check_dissociation(curr_coords):
                    logger.info("Trajectory %d terminated at step %d due to dissociation.", traj_idx, step)
                    all_coords[traj_idx, step:] = curr_coords
                    all_pops[traj_idx, step:] = np.abs(c_amplitudes)**2
                    break
                    
                # Evaluate potential energy surfaces, gradients, and NACVs using AIMNet2-NSE
                V_energies, gradients, d_nacv = self.aimnet2_calc.evaluate_surfaces(curr_coords)
                
                # Electronic Integration
                c_amplitudes = integrate_electronic_rk4(
                    c_amplitudes, V_energies, d_nacv, curr_vels, dt_au
                )
                
                # Decoherence Correction
                E_kin = 0.5 * np.sum(curr_vels**2)
                c_amplitudes = apply_granucci_persico_decoherence(
                    c_amplitudes, curr_state, V_energies, E_kin, dt_au
                )
                
                # Evaluate Hop using seeded RNG
                new_state, hopped = evaluate_hop(
                    c_amplitudes, d_nacv, curr_vels, curr_state, dt_au, rng=rng
                )
                if hopped:
                    delta_V = V_energies[new_state] - V_energies[curr_state]
                    curr_vels, success = rescale_velocity_after_hop(
                        curr_vels, d_nacv[curr_state, new_state], atomic_masses, delta_V
                    )
                    if success:
                        curr_state = new_state
                        
                # Update Nuclear Coordinates using Symplectic Velocity Verlet Scheme
                # A_n = - grad V_curr / M
                grad_curr = gradients[curr_state]
                acc_n = -grad_curr / masses_au[:, np.newaxis]
                
                # Half-step velocity & full-step coordinate
                vel_half = curr_vels + 0.5 * acc_n * dt_au
                curr_coords = curr_coords + vel_half * dt_au
                
                # Evaluate gradients at R_{n+1} for second half-step velocity update
                V_next, grad_next, _ = self.aimnet2_calc.evaluate_surfaces(curr_coords)
                acc_next = -grad_next[curr_state] / masses_au[:, np.newaxis]
                curr_vels = vel_half + 0.5 * acc_next * dt_au
                
        # 4. Compute Vibrational-Averaged Rotational Constants (§5.1 / PULSE-02)
        from .wigner import compute_vibrational_averaged_constants
        flat_frames = all_coords.reshape(-1, n_atoms, 3)
        B_e_mhz, B_0_mhz, rot_meta = compute_vibrational_averaged_constants(flat_frames, atomic_masses)
        
        prov_tag = getattr(self.aimnet2_calc, "provenance_tag", "[E]")
        
        state_chaining_data = {
            "state_in": state_in,
            "state_out": state_out,
            "inherited_geometry_hash": inherited_geometry_hash or "sha256_inherited_geom",
            "inherited_hessian_file": inherited_hessian_file or "inherited_hessian.json"
        }
        
        rotational_constants_data = {
            "B_e_mhz": B_e_mhz,
            "B_0_mhz": B_0_mhz,
            "averaging_method": rot_meta["averaging_method"],
            "provenance_tag": "[D]"
        }

        results = {
            "coordinates": all_coords,
            "velocities": all_vels,
            "populations": all_pops,
            "jobs_dispatched": len(jobs),
            "state_chaining": state_chaining_data,
            "rotational_constants": rotational_constants_data,
            "provenance_tag": prov_tag
        }
        
        # 5. Serialize to HDF5 if output file provided
        if output_h5:
            serialize_swarm_to_hdf5(
                filepath=output_h5,
                swarm_data=results,
                metadata={"LAM_TRIGGER_REQUIRED": False},
                state_chaining=state_chaining_data,
                rotational_constants=rotational_constants_data
            )
            
        return results