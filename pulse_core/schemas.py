from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class StateChainingSchema(BaseModel):
    state_in: str = Field(..., description="Input state identification")
    state_out: str = Field(..., description="Output state identification")
    inherited_geometry_hash: str = Field(..., description="SHA-256 hash of inherited geometry")
    inherited_hessian_file: str = Field(..., description="Path or name of the inherited Hessian file")

class RotationalConstantsSchema(BaseModel):
    B_e_mhz: List[float] = Field(..., description="Equilibrium rotational constants in MHz")
    B_0_mhz: List[float] = Field(..., description="Vibrational-averaged rotational constants in MHz")
    averaging_method: str = Field(..., description="Method used for vibrational averaging")
    provenance_tag: str = Field(..., description="Data provenance tracking tag")

class SwarmResultSchema(BaseModel):
    jobs_dispatched: int = Field(..., description="Number of swarm jobs dispatched")
    state_chaining: StateChainingSchema
    rotational_constants: RotationalConstantsSchema
    provenance_tag: str
