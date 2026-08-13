# **CoChem-PULSE: Non-Adiabatic Photodynamics Engine**

**PI/Developer**: Dr. Joshua John Klaassen
**ORCiD**: [https://orcid.org/0009-0007-1506-4401](https://orcid.org/0009-0007-1506-4401)
**GitHub Organization**: [https://github.com/ProfJJK-CoChem](https://github.com/ProfJJK-CoChem)

> **Important**: CoChem has recently migrated to the **Valeev Stack (MPQC, F12)**. Wavepacket bifurcations and gradient updates are now tightly coupled with MPQC backend derivatives `[E]`.

Please refer to the authoritative [CoChem User Manual](https://github.com/ProfJJK-CoChem/CoChem-BASE/blob/main/CoChem_User_Manual.md) and [Method Matrix](https://github.com/ProfJJK-CoChem/CoChem-BASE/blob/main/Method_Matrix.md) for full execution instructions and basis set provenances.

## **Overview**

**CoChem-PULSE** is the non-adiabatic photodynamics engine of the CoChem suite.

It is responsible for:
- Integrating Tully's Fewest Switches Surface Hopping (FSSH) to track true wavepacket bifurcations at Conical Intersections.
- Deriving classical initial conditions directly from a quantum Wigner sub-sampling of the ground-state Zero-Point Energy `[D]`.
- Simulating explicit time-resolved external laser pulses to actively pump the classical swarm.
- Dispatching massive parallel trajectory swarms to `CoChem-NODE` and profiling femtosecond-resolved electronic state fractional populations.

## **Setup and Installation**

1. Clone PULSE:
   ```bash
   git clone https://github.com/ProfJJK-CoChem/CoChem-PULSE.git
   cd CoChem-PULSE
   ```
2. Link the trajectory runner to your active `CoChem-NODE` environment and ensure MPQC pathing is configured in `pulse_config.json`.

## **Getting Started**

Launch a trajectory swarm via the CLI:
```bash
python pulse_dynamics.py --molecule init.xyz --trajectories 100 --time 500fs
```
Populations and coordinates will stream to the configured `CoChem-NODE` broker instance.
