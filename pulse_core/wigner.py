import hashlib
"""
Wigner Phase-Space Initial Condition Sampling Engine for Quantum Dynamics.
"""

import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Constants in Atomic Units / Physical Constants
HBAR = 1.054571817e-34  # J s
KB = 1.380649e-23        # J / K
AMU_TO_KG = 1.66053906660e-27  # kg
ANGSTROM_TO_M = 1e-10
CM1_TO_RAD_S = 2.99792458e10 * 2.0 * np.pi

def sample_wigner(
    coords: np.ndarray,
    hessian: Optional[np.ndarray] = None,
    n_trajectories: int = 100,
    freqs: Optional[np.ndarray] = None,
    normal_modes: Optional[np.ndarray] = None,
    masses: Optional[np.ndarray] = None,
    temp: float = 300.0,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Wigner quantum phase-space samples (coords and momenta) for initial condition trajectory ensembles.
    
    Args:
        coords: Reference geometry array of shape (N_atoms, 3) in Angstroms.
        hessian: Cartesian mass-weighted Hessian array of shape (3N, 3N) in Hartree/Bohr^2 or J/(kg m^2).
        n_trajectories: Number of trajectory initial conditions to sample.
        freqs: Optional 1D array of vibrational frequencies in cm^-1.
        normal_modes: Optional normal mode transformation matrix of shape (3N, 3N).
        masses: 1D array of atomic masses in amu (length N_atoms). Defaults to carbon (12.0) if None.
        temp: Temperature in Kelvin (default 300.0).
        random_seed: Optional seed for reproducible sampling.
        
    Returns:
        Tuple (sampled_coords, sampled_momenta) each of shape (n_trajectories, N_atoms, 3).
    """
    rng = np.random.default_rng(random_seed)
        
    coords = np.asarray(coords, dtype=float)
    n_atoms = coords.shape[0]
    n_dof = 3 * n_atoms
    
    if masses is None:
        masses = np.full(n_atoms, 12.0)
    else:
        masses = np.asarray(masses, dtype=float)
        
    # Construct mass vector for 3N DOF
    m_3n = np.repeat(masses, 3)  # amu
    m_3n_kg = m_3n * AMU_TO_KG
    
    if freqs is None or normal_modes is None:
        if hessian is None:
            # Build an acoustic harmonic approximation if no Hessian is provided
            logger.warning("No Hessian provided; creating simple harmonic approximation based on coordinates.")
            hessian = np.eye(n_dof) * 0.1
            
        # Mass-weight Hessian: H_mw = M^{-1/2} H M^{-1/2}
        inv_sqrt_m = 1.0 / np.sqrt(m_3n_kg)
        mw_hessian = hessian * np.outer(inv_sqrt_m, inv_sqrt_m)
        
        evals, normal_modes = np.linalg.eigh(mw_hessian)
        # Filter negative / small eigenvalues (rotations/translations)
        evals_pos = np.maximum(evals, 1e-8)
        omega = np.sqrt(evals_pos)  # rad/s approx
    else:
        omega = freqs * CM1_TO_RAD_S
        normal_modes = np.asarray(normal_modes, dtype=float)
        
    sampled_coords = np.zeros((n_trajectories, n_atoms, 3))
    sampled_momenta = np.zeros((n_trajectories, n_atoms, 3))
    
    beta = 1.0 / (KB * max(temp, 1.0))
    
    for i in range(n_trajectories):
        q_mode = np.zeros(n_dof)
        p_mode = np.zeros(n_dof)
        
        for k in range(n_dof):
            w_k = max(omega[k], 1e-4)
            # Quantum Wigner variances: sigma_q^2 = (hbar / (2 w)) coth(beta hbar w / 2)
            x_k = HBAR * w_k * beta / 2.0
            coth_k = 1.0 / np.tanh(min(x_k, 50.0))
            
            sigma_q = np.sqrt((HBAR / (2.0 * w_k)) * coth_k)  # mass-weighted coordinate variance
            sigma_p = np.sqrt((HBAR * w_k / 2.0) * coth_k)     # mass-weighted momentum variance
            
            # Exact Quantum Harmonic Oscillator Wigner probability distribution sampling via Box-Muller transform
            u1 = float(rng.uniform(1e-12, 1.0))
            u2 = float(rng.uniform(0.0, 1.0))
            mag = np.sqrt(-2.0 * np.log(u1))
            q_mode[k] = mag * np.cos(2.0 * np.pi * u2) * sigma_q
            p_mode[k] = mag * np.sin(2.0 * np.pi * u2) * sigma_p
            
        # Transform back to Cartesian coordinates & momenta
        dq_cart_mw = normal_modes @ q_mode  # mass-weighted displacements
        dp_cart_mw = normal_modes @ p_mode  # mass-weighted momenta
        
        # Un-mass-weight: dq = dq_mw / sqrt(m), dp = dp_mw * sqrt(m)
        dq_cart = (dq_cart_mw * inv_sqrt_m).reshape(n_atoms, 3) / ANGSTROM_TO_M
        dp_cart = (dp_cart_mw / inv_sqrt_m).reshape(n_atoms, 3)  # kg m/s
        
        sampled_coords[i] = coords + dq_cart
        sampled_momenta[i] = dp_cart
        
    logger.info("Sampled %d Wigner initial conditions for %d atoms at T=%.1f K.", n_trajectories, n_atoms, temp)
    return sampled_coords, sampled_momenta


def compute_inertia_tensor(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Computes 3x3 inertia tensor I (in amu Angstrom^2) for 3D coordinates (Angstroms) and masses (amu).
    Translates center of mass to origin before computing moment of inertia.
    """
    coords = np.asarray(coords, dtype=float)
    masses = np.asarray(masses, dtype=float)
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / np.sum(masses)
    r = coords - com
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    Ixx = np.sum(masses * (y**2 + z**2))
    Iyy = np.sum(masses * (x**2 + z**2))
    Izz = np.sum(masses * (x**2 + y**2))
    Ixy = -np.sum(masses * x * y)
    Ixz = -np.sum(masses * x * z)
    Iyz = -np.sum(masses * y * z)
    return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]], dtype=float)


def safe_invert_inertia_tensor(I: np.ndarray, tol_rel: float = 1e-4) -> np.ndarray:
    """
    Safely invert a 3x3 moment of inertia tensor I using spectral eigendecomposition.
    Zeroes out axial/singular eigenvalues where lambda_k < tol_rel * lambda_max.
    Prevents catastrophic floating-point overflow for near-singular or linear rotors.

    Args:
        I: Symmetric 3x3 inertia tensor array (amu Angstrom^2).
        tol_rel: Relative tolerance threshold for singular eigenvalues (default 1e-4).

    Returns:
        3x3 inverse moment of inertia tensor mu = I^-1 (amu^-1 Angstrom^-2).
    """
    eigvals, V = np.linalg.eigh(I)
    eig_max = np.max(eigvals)
    if eig_max <= 0:
        return np.zeros_like(I)
    inv_eig = np.zeros_like(eigvals)
    mask = eigvals >= tol_rel * eig_max
    inv_eig[mask] = 1.0 / eigvals[mask]
    return V @ np.diag(inv_eig) @ V.T


def compute_vibrational_averaged_constants(
    frames_coords: np.ndarray,
    masses: np.ndarray,
    tol_rel: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Computes B_e (from mean geometry) and B_0 (from frame-wise inverse inertia tensor average).
    Strictly satisfies Jensen's inequality: <1/I> >= 1/<I> (§5.1).
    
    Conversion constant: h / (8 * pi^2 * amu * Angstrom^2) * 1e-6 = 505379.006 MHz amu Angstrom^2.
    
    Args:
        frames_coords: 3D array of shape (N_frames, N_atoms, 3) or 2D single frame (N_atoms, 3).
        masses: 1D array of atomic masses in amu.
        tol_rel: Relative tolerance threshold for singular/axial eigenvalue inversion (default 1e-4).
        
    Returns:
        Tuple (B_e_mhz, B_0_mhz, metadata):
            - B_e_mhz: 1D array of equilibrium rotational constants [A_e, B_e, C_e] in MHz (descending).
            - B_0_mhz: 1D array of zero-point vibrational ground-state constants [A_0, B_0, C_0] in MHz (descending).
            - metadata: Dictionary containing averaging_method, provenance_tag, and jensen_satisfied flag.
    """
    CONV_MHZ = 505379.006
    frames = np.asarray(frames_coords, dtype=float)
    if frames.ndim == 2:
        frames = frames[np.newaxis, :, :]
        
    masses = np.asarray(masses, dtype=float)
    
    # 1. Compute B_e from equilibrium/mean geometry <R>
    mean_coords = np.mean(frames, axis=0)
    I_eq = compute_inertia_tensor(mean_coords, masses)
    eigvals_eq = np.sort(np.linalg.eigvalsh(I_eq))  # ascending [I_a, I_b, I_c]
    eigvals_eq_pos = np.maximum(eigvals_eq, 1e-10)
    B_e_mhz = np.sort(CONV_MHZ / eigvals_eq_pos)[::-1]  # descending [A_e, B_e, C_e]
    
    # 2. Compute frame-wise principal moments of inertia and inverse principal moments
    I_eig_list = []
    mu_eig_list = []
    active_masks = []

    for frame in frames:
        I_t = compute_inertia_tensor(frame, masses)
        eig_t = np.sort(np.linalg.eigvalsh(I_t))  # ascending [I_a, I_b, I_c]
        eig_max_t = np.max(eig_t)

        mu_t = np.zeros(3)
        mask_t = eig_t >= tol_rel * eig_max_t
        active_masks.append(mask_t)

        if np.any(mask_t):
            mu_t[mask_t] = 1.0 / eig_t[mask_t]

        I_eig_list.append(eig_t)
        mu_eig_list.append(mu_t)

    # 3. Inverse of mean inertia tensor (<I>)^-1 for Jensen's inequality reference
    I_avg_eig = np.mean(I_eig_list, axis=0)  # ascending [<I_a>, <I_b>, <I_c>]
    eig_max_avg = np.max(I_avg_eig)

    active_in_all = np.all(active_masks, axis=0)
    mask_avg = (I_avg_eig >= tol_rel * eig_max_avg) & active_in_all

    inv_I_avg_eig = np.zeros(3)
    if np.any(mask_avg):
        inv_I_avg_eig[mask_avg] = 1.0 / I_avg_eig[mask_avg]

    B_inv_I_avg = np.sort(CONV_MHZ * inv_I_avg_eig)[::-1]  # descending [A_inv, B_inv, C_inv]

    # 4. Frame-wise inverse inertia average <I^-1> (B_0)
    mu_avg_eig = np.mean(mu_eig_list, axis=0)
    B_0_mhz = np.sort(CONV_MHZ * mu_avg_eig)[::-1]  # descending [A_0, B_0, C_0]

    # 5. Verify Jensen's Inequality compliance: <I^-1> >= (<I>)^-1 (§5.1)
    jensen_satisfied = bool(np.all(B_0_mhz >= B_inv_I_avg - 1e-6))

    meta = {
        "averaging_method": "INVERSE_TENSOR_AVERAGE_JENSEN",
        "provenance_tag": "[D]",
        "jensen_satisfied": jensen_satisfied
    }
    return B_e_mhz, B_0_mhz, meta


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