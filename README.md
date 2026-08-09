# CoChem-PULSE

**CoChem-PULSE** is the Non-Adiabatic Photodynamics engine of the extended CoChem suite.

It is responsible for:
- Integrating Tully's Fewest Switches Surface Hopping (FSSH) to track true wavepacket bifurcations at Conical Intersections.
- Deriving classical initial conditions strictly from a quantum Wigner sub-sampling of the ground-state Zero-Point Energy.
- Simulating explicit time-resolved external laser pulses to actively pump the classical swarm.
- Dispatching massive parallel trajectory swarms to `CoChem-NODE` and tracking femtosecond-resolved electronic state fractional populations.

## Usage
Please refer to the authoritative `CoChem_Master_User_Manual.md` located in the `CoChem-BASE` repository for full execution instructions across the entire pipeline.