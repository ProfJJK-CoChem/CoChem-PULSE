from typing import Any, Dict, List, Optional
"""
Unit tests for Wigner Phase-Space Initial Condition Sampling Engine.
"""

import pytest
import numpy as np
from pulse_core.wigner import sample_wigner

def test_wigner_sampling_shapes_and_types() -> None:
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.1],
        [0.0, 1.0, -0.3]
    ])
    
    n_trajectories = 15
    q_sampled, p_sampled = sample_wigner(
        coords,
        n_trajectories=n_trajectories,
        temp=300.0,
        random_seed=42
    )
    
    assert q_sampled.shape == (n_trajectories, 3, 3)
    assert p_sampled.shape == (n_trajectories, 3, 3)
    assert not np.isnan(q_sampled).any()
    assert not np.isnan(p_sampled).any()

def test_wigner_variance_non_zero() -> None:
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    
    q_sampled, p_sampled = sample_wigner(
        coords,
        n_trajectories=50,
        temp=300.0,
        random_seed=123
    )
    
    # Check that sampling produces non-zero variance around reference coords
    q_std = np.std(q_sampled, axis=0)
    p_std = np.std(p_sampled, axis=0)
    
    assert np.all(q_std > 0.0)
    assert np.all(p_std > 0.0)