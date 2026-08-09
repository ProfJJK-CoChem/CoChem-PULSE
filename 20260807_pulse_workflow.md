# CoChem-PULSE: Execution Workflow (2026-08-07)

## Phase 1: Swarm Initialization
1. **Wigner Sampling:** PULSE pulls the ground-state Hessian from TOPOS/TORQ and generates 100+ discrete starting geometries and momentum vectors.
2. **Pump Pulse Simulation:** The system applies a semi-classical oscillating electric field (specified by the user in nm and FWHM duration) to "excite" the swarm to the $S_1$ surface.

## Phase 2: Trajectory Propagation
1. **RK4 Integration:** The electronic amplitude is integrated using RK4 at an extreme sub-femtosecond timestep ($0.001$ fs), while nuclei step at $0.5$ fs.
2. **Surface Hopping:** If a trajectory crosses a Conical Intersection, it evaluates the hopping probability. If successful, velocities are rescaled along the NACV. Frustrated hops are flagged and reversed.
3. **Kill-Switch:** Any trajectory resulting in atomic dissociation ($>10$ Å bond length) is instantly terminated to save CPU cycles.

## Phase 3: Reporting & UX
1. **HTML 3D Playback:** The Jupyter UI renders a 3D Py3Dmol animation of the molecule, flashing when a surface hop occurs.
2. **Population Plots:** A real-time Plotly graph displays the fraction of trajectories in $S_2$, $S_1$, and $S_0$ over time (fs), directly yielding the theoretical quantum yield.
3. **Provenance Logging:** The random seeds, explicit velocities, and failed trajectories are cryptographically serialized to `cochem_state.h5`.
