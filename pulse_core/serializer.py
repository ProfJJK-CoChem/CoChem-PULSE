"""
HDF5 Provenance and Trajectory Swarm Serialization Engine for PULSE.
"""

import logging
import time
from typing import Dict, Any, Optional
import numpy as np
import h5py

logger = logging.getLogger(__name__)

def serialize_swarm_to_hdf5(
    filepath: str,
    swarm_data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Serializes swarm trajectory histories, velocities, populations, and hopping logs into HDF5 state format.
    
    Args:
        filepath: Target HDF5 file path (e.g. cochem_state.h5).
        swarm_data: Dictionary containing:
            - 'coordinates': 4D array (n_trajectories, n_frames, n_atoms, 3)
            - 'velocities': 4D array (n_trajectories, n_frames, n_atoms, 3)
            - 'populations': 3D array (n_trajectories, n_frames, n_states)
            - 'hopping_events': List or 2D array of hop records
            - 'random_seeds': 1D array of trajectory seeds
        metadata: Additional metadata dictionary for dataset attributes.
    """
    with h5py.File(filepath, "a") as h5:
        group_path = "pulse/trajectories"
        if group_path in h5:
            del h5[group_path]
            
        grp = h5.create_group(group_path)
        
        for key in ["coordinates", "velocities", "populations", "random_seeds"]:
            if key in swarm_data and swarm_data[key] is not None:
                data_arr = np.asarray(swarm_data[key])
                grp.create_dataset(key, data=data_arr, compression="gzip")
                
        if "hopping_events" in swarm_data and swarm_data["hopping_events"] is not None:
            hop_data = np.asarray(swarm_data["hopping_events"])
            grp.create_dataset("hopping_events", data=hop_data)
            
        # Metadata attributes (PULSE-17 & Base/Topos matrix standard)
        meta = metadata or {}
        grp.attrs["LAM_TRIGGER_REQUIRED"] = meta.get("LAM_TRIGGER_REQUIRED", False)
        grp.attrs["symmetry_group"] = meta.get("symmetry_group", "C1")
        grp.attrs["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        grp.attrs["version"] = meta.get("version", "1.0.0")
        
    logger.info("Successfully serialized swarm trajectory data to HDF5 file: %s", filepath)
