import json
import logging
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

def _load_system_config() -> Dict[str, Any]:
    """Load system configuration from registry or environment path."""
    config_env = os.environ.get("COCHEM_SYSTEM_CONFIG")
    config_paths = []
    if config_env:
        config_paths.append(Path(config_env))
    config_paths.extend([
        Path(__file__).resolve().parents[2] / "cochem_system_config.json",
        Path("cochem_system_config.json"),
        Path("cochem_setup/cochem_system_config.json"),
        Path.home() / ".cochem" / "cochem_system_config.json"
    ])
    
    for path in config_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load config from %s: %s", path, e)
    return {}

def dispatch_swarm_to_node(
    wigner_seeds: Tuple[np.ndarray, np.ndarray],
    engine: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    use_mps: bool = True,
    thread_percentage: int = 25
) -> List[Dict[str, Any]]:
    """
    Formats 100+ separate execution payloads and hands them to NODE with the [CPU_DIST] and [CUDA_MPS] tags.
    Reinstates AIMNet2-NSE / sTDA / FOMO-CI for Non-Adiabatic Surface Hopping (NASH) molecular dynamics.
    
    Args:
        wigner_seeds: Tuple of (coords, momenta) arrays.
        engine: Dynamics engine identifier ("AIMNet2-NSE", "sTDA", "FOMO-CI"). If None, resolved from config.
        config: Optional configuration dictionary override.
        use_mps: Whether CUDA MPS co-scheduling is enabled.
        thread_percentage: GPU thread percentage per worker process.
        
    Returns:
        List of formatted job payload dictionaries.
    """
    # PULSE-01: Validate tuple input and numpy dimensions
    if not isinstance(wigner_seeds, tuple) or len(wigner_seeds) != 2:
        raise TypeError("wigner_seeds must be a 2-tuple of (coordinates, momenta) arrays.")
    
    coords, momenta = wigner_seeds
    if coords is None or momenta is None:
        raise ValueError("Coordinates and momenta in wigner_seeds cannot be None.")
    
    coords = np.asarray(coords)
    momenta = np.asarray(momenta)
    
    if coords.ndim != 3:
        raise ValueError(f"Coordinates array must have 3 dimensions (n_samples, n_atoms, 3), got {coords.ndim}D.")
    if momenta.ndim != 3:
        raise ValueError(f"Momenta array must have 3 dimensions (n_samples, n_atoms, 3), got {momenta.ndim}D.")
    
    # PULSE-02: Verify array length equality
    if len(coords) != len(momenta):
        raise ValueError(f"Array length mismatch: len(coords)={len(coords)} != len(momenta)={len(momenta)}.")
    
    # PULSE-19 & PULSE-09 & PULSE-11: Resolve engine selection from config or registry
    sys_config = config or _load_system_config()
    selected_engine = engine or sys_config.get("pulse", {}).get("engine") or sys_config.get("default_engine") or "AIMNet2-NSE"
    
    allowed_engines = {"AIMNet2-NSE", "sTDA", "FOMO-CI"}
    if selected_engine not in allowed_engines:
        logger.warning("Engine '%s' not in standard set %s, proceeding with configured selection.", selected_engine, allowed_engines)
    
    jobs = []
    for i in range(len(coords)):
        # PULSE-18: Check for NaNs/Infs and safely handle .tolist()
        try:
            if np.isnan(coords[i]).any() or np.isinf(coords[i]).any():
                raise ValueError(f"Trajectory {i} contains NaN or Inf values in coordinates.")
            if np.isnan(momenta[i]).any() or np.isinf(momenta[i]).any():
                raise ValueError(f"Trajectory {i} contains NaN or Inf values in momenta.")
            
            coord_list = coords[i].tolist()
            momenta_list = momenta[i].tolist()
        except Exception as e:
            logger.error("Failed to convert trajectory %d arrays to list: %s", i, e)
            raise ValueError(f"Invalid numeric data in trajectory {i}: {e}") from e
        
        job = {
            "job_id": f"traj_{i}",
            "coordinates": coord_list,
            "momenta": momenta_list,
            "engine": selected_engine,
            "tags": ["[CPU_DIST]", "[NASH]", "[CUDA_MPS]"],
            "mps_config": {
                "use_mps": use_mps,
                "thread_percentage": thread_percentage
            }
        }
        jobs.append(job)
        
    # PULSE-10: Use structured logging instead of raw print
    logger.info("Dispatched %d NASH %s trajectories to swarm with CUDA MPS co-scheduling.", len(jobs), selected_engine)
    return jobs
