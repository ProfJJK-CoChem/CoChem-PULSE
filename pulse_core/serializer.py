import hashlib
"""
HDF5 Provenance and Trajectory Swarm Serialization Engine for PULSE.
"""

import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import h5py

logger = logging.getLogger(__name__)

def serialize_swarm_to_hdf5(
    filepath: str,
    swarm_data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    state_chaining: Optional[Dict[str, Any]] = None,
    rotational_constants: Optional[Dict[str, Any]] = None
) -> None:
    """
    Serializes swarm trajectory histories, state chaining pipeline fields,
    rotational constants (Jensen inverse inertia average), and provenance metadata into HDF5.
    
    Args:
        filepath: Target HDF5 file path (e.g. cochem_state.h5).
        swarm_data: Dictionary containing coordinates, velocities, populations, hopping_events, etc.
        metadata: General swarm metadata attributes.
        state_chaining: Dict containing state_in, state_out, inherited_geometry_hash, inherited_hessian_file.
        rotational_constants: Dict containing B_e_mhz, B_0_mhz, averaging_method, provenance_tag.
    """
    with h5py.File(filepath, "a") as h5:
        group_path = "pulse/trajectories"
        if group_path in h5:
            del h5[group_path]
            
        grp = h5.create_group(group_path)
        
        # 1. Trajectory Datasets
        for key in ["coordinates", "velocities", "populations", "random_seeds"]:
            if key in swarm_data and swarm_data[key] is not None:
                data_arr = np.asarray(swarm_data[key])
                ds = grp.create_dataset(key, data=data_arr, compression="gzip")
                if key == "populations":
                    ds.attrs["provenance_tag"] = swarm_data.get("provenance_tag", "[D]")
                
        if "hopping_events" in swarm_data and swarm_data["hopping_events"] is not None:
            hop_data = np.asarray(swarm_data["hopping_events"])
            hop_ds = grp.create_dataset("hopping_events", data=hop_data)
            hop_ds.attrs["provenance_tag"] = swarm_data.get("provenance_tag", "[D]")

        # 2. State-Chaining Pipeline Subgroup (§8B / PULSE-03)
        sc_grp = grp.create_group("state_chaining")
        sc_info = state_chaining or swarm_data.get("state_chaining", {})
        sc_grp.attrs["state_in"] = sc_info.get("state_in", "T2-3h_r2SCAN-3c_TightOpt")
        sc_grp.attrs["state_out"] = sc_info.get("state_out", "T4-1d_FSSH_Trajectory")
        sc_grp.attrs["inherited_geometry_hash"] = sc_info.get("inherited_geometry_hash", "sha256_none")
        sc_grp.attrs["inherited_hessian_file"] = sc_info.get("inherited_hessian_file", "none")
        
        # 3. Rotational Constants Subgroup (§5.1 / PULSE-03)
        rc_grp = grp.create_group("rotational_constants")
        rc_info = rotational_constants or swarm_data.get("rotational_constants", {})
        if "B_e_mhz" in rc_info and rc_info["B_e_mhz"] is not None:
            rc_grp.create_dataset("B_e_mhz", data=np.asarray(rc_info["B_e_mhz"]))
        if "B_0_mhz" in rc_info and rc_info["B_0_mhz"] is not None:
            rc_grp.create_dataset("B_0_mhz", data=np.asarray(rc_info["B_0_mhz"]))
        rc_grp.attrs["averaging_method"] = rc_info.get("averaging_method", "INVERSE_TENSOR_AVERAGE_JENSEN")
        rc_grp.attrs["provenance_tag"] = rc_info.get("provenance_tag", "[D]")

        # 4. Global Metadata Attributes
        meta = metadata or {}
        grp.attrs["LAM_TRIGGER_REQUIRED"] = meta.get("LAM_TRIGGER_REQUIRED", False)
        grp.attrs["symmetry_group"] = meta.get("symmetry_group", "C1")
        grp.attrs["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        grp.attrs["version"] = meta.get("version", "4.0.0")
        grp.attrs["provenance_tag"] = swarm_data.get("provenance_tag", "[D]")
        
    logger.info("Successfully serialized swarm trajectory data, state chaining, and rotational constants to HDF5: %s", filepath)
    
    # Calculate and log tracking checksum
    try:
        artifact_hash = calculate_artifact_sha256(filepath)
        logger.info("Artifact Checksum (SHA-256): %s", artifact_hash)
    except Exception as e:
        logger.warning("Failed to calculate artifact tracking hash: %s", e)
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