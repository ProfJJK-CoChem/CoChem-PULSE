"""
CoChem-PULSE: Non-Adiabatic Surface Hopping (NASH) & Quantum Trajectory Dynamics Package
"""

from .swarm_manager import dispatch_swarm_to_node
from .wigner import sample_wigner
from .fssh_rk4 import integrate_electronic_rk4, evaluate_hop
from .dispatcher import PulseDispatcher
from .laser import simulate_pump_pulse
from .nacv import compute_nacv, rescale_velocity_after_hop
from .decoherence import apply_granucci_persico_decoherence
from .kill_switch import check_dissociation
from .viz import plot_population_dynamics, render_trajectory_3d
from .serializer import serialize_swarm_to_hdf5

from .aimnet2_nse import AIMNet2NSECalculator

__all__ = [
    "dispatch_swarm_to_node",
    "sample_wigner",
    "integrate_electronic_rk4",
    "evaluate_hop",
    "PulseDispatcher",
    "AIMNet2NSECalculator",
    "simulate_pump_pulse",
    "compute_nacv",
    "rescale_velocity_after_hop",
    "apply_granucci_persico_decoherence",
    "check_dissociation",
    "plot_population_dynamics",
    "render_trajectory_3d",
    "serialize_swarm_to_hdf5",
]
