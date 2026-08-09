# CoChem-PULSE: Software Engineering Specification
**Target Phase:** Python Implementation

This document serves as the exact coding blueprint for the next LLM agent to construct the `CoChem-PULSE` repository.

## 1. Directory & File Architecture
```text
CoChem-PULSE/
├── pulse_core/
│   ├── __init__.py
│   ├── dispatcher.py      # Entry point for BASE payload ingestion
│   ├── wigner.py          # Quantum sub-sampling of initial conditions
│   ├── fssh_rk4.py        # Tully Surface Hopping electronic integration
│   └── swarm_manager.py   # Mass-dispatch of independent trajectories
├── tests/
│   ├── test_wigner.py
│   └── test_rk4.py
├── requirements.txt       # numba, jax, h5py, numpy, scipy
└── README.md
```

## 2. File-by-File Blueprint

### `pulse_core/wigner.py`
- **Purpose:** Mimics quantum zero-point uncertainty to seed the swarm.
- **Functions:**
  - `def sample_wigner(hessian: np.ndarray, n_trajectories: int) -> tuple:`
    - *Returns:* Two arrays `(coords_swarm, momenta_swarm)` representing $N$ unique starting configurations.

### `pulse_core/fssh_rk4.py`
- **Purpose:** Integrates the electronic wavepacket and evaluates hops.
- **Functions:**
  - `def integrate_electronic_rk4(c_amplitudes: np.ndarray, hamiltonian: np.ndarray, dt: float) -> np.ndarray:`
    - *Returns:* Updated complex amplitudes for the diabatic states.
  - `def evaluate_hop(c_amplitudes, nacv, momenta) -> bool:`
    - *Returns:* `True` if a surface hop is allowed by energy conservation.

### `pulse_core/swarm_manager.py`
- **Purpose:** Manages the massively parallel trajectories.
- **Functions:**
  - `def dispatch_swarm_to_node(wigner_seeds: tuple):`
    - *Action:* Formats 100+ separate execution payloads and hands them to NODE with the `[CPU_DIST]` tag.

## 3. Execution Data Flow (The Payload Trace)
1. **Payload Ingest:** `dispatcher.py` receives a payload triggered by LUMOS identifying a Conical Intersection.
2. **Wigner Initialization:** Parses the ground-state Hessian from `/topos/mace_ensemble`. `wigner.py` generates $100+$ unique momentum/coordinate seeds.
3. **Dispatch:** `swarm_manager.py` loops over the seeds, formatting independent jobs to hit the `sTDA` fast-integration engine via NODE.
4. **Integration Loop:** For each trajectory, `fssh_rk4.py` integrates the electronic amplitudes at $0.001$ fs, checking for hops at $0.5$ fs nuclear steps.
5. **Decoherence & Logging:** Frustrated hops are reversed. Granucci-Persico decoherence is applied.
6. **Serialization:** When the swarm finishes, the fractional state populations over time and the atomic coordinates are written to `/pulse/wigner_swarm/`.

## 4. PyTest Roadmap
- **Test 1 (`test_rk4.py`):** Provide a static $2 \times 2$ Hamiltonian matrix. Assert that `integrate_electronic_rk4` strictly conserves the norm ($\sum |c_i|^2 = 1.0$) over 10,000 steps.
- **Test 2 (`test_wigner.py`):** Assert that a sufficiently large `n_trajectories` (e.g., 5000) reproduces the classical harmonic oscillator variance corresponding to the zero-point energy.
