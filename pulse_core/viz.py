"""
Visualization Engine for Quantum Trajectory Dynamics and Electronic State Populations.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

def plot_population_dynamics(
    time_grid: np.ndarray,
    state_populations: np.ndarray,
    state_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generates fractional electronic state population data / Plotly dictionary spec over time.
    
    Args:
        time_grid: 1D array of time steps (fs).
        state_populations: 2D array of shape (n_times, n_states) with population fractions.
        state_names: Optional list of state names (e.g. ["S0", "S1", "S2"]).
        
    Returns:
        Plotly-compatible dictionary specification for rendering population curves.
    """
    time_grid = np.asarray(time_grid, dtype=float)
    pops = np.asarray(state_populations, dtype=float)
    
    n_times, n_states = pops.shape
    if state_names is None:
        state_names = [f"S_{i}" for i in range(n_states)]
        
    traces = []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    for s in range(n_states):
        color = colors[s % len(colors)]
        trace = {
            "x": time_grid.tolist(),
            "y": pops[:, s].tolist(),
            "mode": "lines",
            "name": state_names[s],
            "line": {"color": color, "width": 2}
        }
        traces.append(trace)
        
    layout = {
        "title": "Electronic State Fractional Population Dynamics",
        "xaxis": {"title": "Time (fs)"},
        "yaxis": {"title": "Population Fraction", "range": [0, 1.05]},
        "template": "plotly_white"
    }
    
    return {"data": traces, "layout": layout}

def render_trajectory_3d(
    coords_history: np.ndarray,
    atomic_symbols: Optional[List[str]] = None
) -> str:
    """
    Renders 3D trajectory frames into XYZ format or Py3Dmol animation string.
    
    Args:
        coords_history: 3D array of shape (n_frames, n_atoms, 3) in Angstroms.
        atomic_symbols: List of element symbols of length n_atoms.
        
    Returns:
        Multi-frame XYZ string suitable for Py3Dmol / WebGL rendering.
    """
    coords = np.asarray(coords_history, dtype=float)
    n_frames, n_atoms, _ = coords.shape
    
    if atomic_symbols is None:
        atomic_symbols = ["C"] * n_atoms
        
    xyz_frames = []
    for f in range(n_frames):
        lines = [str(n_atoms), f"Frame {f}"]
        for a in range(n_atoms):
            sym = atomic_symbols[a]
            x, y, z = coords[f, a]
            lines.append(f"{sym:<2s} {x:12.6f} {y:12.6f} {z:12.6f}")
        xyz_frames.append("\n".join(lines))
        
    full_xyz = "\n".join(xyz_frames)
    logger.info("Rendered %d frames of 3D trajectory animation.", n_frames)
    return full_xyz