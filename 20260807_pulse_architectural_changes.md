# CoChem-PULSE: Architectural Changes (2026-08-07)

## 1. Tully's Fewest Switches Surface Hopping (FSSH)
**Target File:** `pulse_core/fssh.py`
**Required Architectural Change:**
- PULSE must completely abandon static state evaluations and implement FSSH to integrate true non-adiabatic wavepacket propagation. Energy-based decoherence corrections (e.g., Granucci-Persico) must be strictly enforced.

## 2. Wigner Sub-Sampling
**Target File:** `pulse_core/wigner.py`
**Required Architectural Change:**
- Initial coordinates and momenta for the trajectory swarm must be sampled from a quantum Wigner distribution derived from the ground-state Zero-Point Energy, ensuring true quantum uncertainty dictates the starting classical conditions.

## 3. High-Speed Semi-Empirical Integration
**Target File:** `pulse_core/swarm_manager.py`
**Required Architectural Change:**
- To process 100+ trajectories within the Time-Tier, PULSE must utilize ultra-fast sTDA or FOMO-CI. Non-adiabatic coupling vectors (NACVs) must be computed on-the-fly, instantly rescaling atomic velocities upon a hop to conserve energy.
