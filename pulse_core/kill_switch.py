"""
Trajectory Dissociation Kill-Switch Monitor for Trajectory Swarms.
"""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

def check_dissociation(
    coords: np.ndarray,
    max_dist: float = 10.0,
    min_dist: float = 0.4
) -> bool:
    """
    Checks if a molecular trajectory has undergone fragment dissociation or non-physical atom overlap.
    
    Args:
        coords: Cartesian coordinates array of shape (N_atoms, 3) in Angstroms.
        max_dist: Maximum allowable pairwise interatomic distance before triggering kill-switch (Angstroms).
        min_dist: Minimum allowable pairwise distance (detects catastrophic collapse/fusion).
        
    Returns:
        True if trajectory should be killed (dissociated or collapsed), False otherwise.
    """
    coords = np.asarray(coords, dtype=float)
    n_atoms = coords.shape[0]
    
    if n_atoms <= 1:
        return False
        
    # Pairwise distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    
    # Check for unphysical atomic fusion (min distance)
    np.fill_diagonal(dist_matrix, 999.0)
    min_r = np.min(dist_matrix)
    if min_r < min_dist:
        logger.warning("Kill-switch triggered: atomic collapse detected (min r = %.3f A < %.3f A).", min_r, min_dist)
        return True
        
    # Check if any atom has detached from all other atoms beyond max_dist
    min_dist_per_atom = np.min(dist_matrix, axis=1)
    if np.any(min_dist_per_atom > max_dist):
        detached_atoms = np.where(min_dist_per_atom > max_dist)[0]
        logger.info("Kill-switch triggered: detached atom(s) %s beyond max_dist %.1f A.", list(detached_atoms), max_dist)
        return True
        
    return False