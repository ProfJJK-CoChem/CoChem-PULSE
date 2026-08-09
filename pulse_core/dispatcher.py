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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
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
        output_h5: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes complete surface hopping trajectory ensemble calculation.
        """
        ref_coords = np.asarray(ref_coords, dtype=float)
        n_atoms = len(ref_coords)
        
        logger.info("Starting PULSE trajectory swarm pipeline for %d atoms (%d trajectories, %d steps).", n_atoms, n_trajectories, n_steps)
        
        # 1. Wigner Phase-Space Sampling
        coords_sampled, momenta_sampled = sample_wigner(
            coords=ref_coords,
            n_trajectories=n_trajectories,
            temp=temp_k
        )
        
        # 2. Swarm Job Preparation
        jobs = dispatch_swarm_to_node(
            wigner_seeds=(coords_sampled, momenta_sampled),
            config=self.config
        )
        
        # 3. Simulate Trajectory Swarms (Local or Node integration)
        dt_au = dt_fs * 41.34137  # fs to atomic time units
        n_states = 3
        
        all_coords = np.zeros((n_trajectories, n_steps, n_atoms, 3))
        all_vels = np.zeros((n_trajectories, n_steps, n_atoms, 3))
        all_pops = np.zeros((n_trajectories, n_steps, n_states))
        
        for traj_idx in range(n_trajectories):
            curr_coords = coords_sampled[traj_idx].copy()
            curr_vels = momenta_sampled[traj_idx].copy() / 12.0  # mass approx
            
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
                
                # Evaluate Hop
                new_state, hopped = evaluate_hop(
                    c_amplitudes, d_nacv, curr_vels, curr_state, dt_au
                )
                if hopped:
                    delta_V = V_energies[new_state] - V_energies[curr_state]
                    masses = np.full(n_atoms, 12.0)
                    curr_vels, success = rescale_velocity_after_hop(
                        curr_vels, d_nacv[curr_state, new_state], masses, delta_V
                    )
                    if success:
                        curr_state = new_state
                        
                # Update Nuclear Coordinates (Velocity Verlet / Euler step)
                curr_coords += curr_vels * dt_au * 0.05
                
        results = {
            "coordinates": all_coords,
            "velocities": all_vels,
            "populations": all_pops,
            "jobs_dispatched": len(jobs)
        }
        
        # 4. Serialize to HDF5 if output file provided
        if output_h5:
            serialize_swarm_to_hdf5(output_h5, results, {"LAM_TRIGGER_REQUIRED": False})
            
        return results
