import streamlit as st
import subprocess
import os
import sys
import psutil
import atexit
import logging
from pathlib import Path
from pydantic import BaseModel, Field

st.set_page_config(page_title="CoChem-PULSE - Native Pipeline UI", layout="wide")
logger = logging.getLogger(__name__)

class PulseConfig(BaseModel):
    target_smiles: str = Field(..., description="Target SMILES string")
    run_mode: str = Field(..., description="Execution Mode (Fast or Accurate)")
    output_dir: Path = Field(default_factory=Path.home)

def kill_zombie_processes() -> None:
    target_procs = ['orca', 'xtb', 'mpi', 'crest']
    for proc in psutil.process_iter(['name']):
        try:
            name = proc.info['name'].lower()
            if any(target in name for target in target_procs):
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            raise NotImplementedError("Implementation pending")
atexit.register(kill_zombie_processes)

st.title("🔬 CoChem-PULSE Control Panel")
st.markdown("This UI executes raw, heavy mathematical payloads natively.")

with st.sidebar:
    st.header("Pipeline Configuration")
    target_smiles = st.text_input("Target SMILES", "CCO")
    run_mode = st.selectbox("Execution Mode", ["Fast", "Accurate"])

if st.button("🚀 Execute Default Pipeline"):
    with st.spinner(f"Triggering quantum physics executor for {target_smiles}..."):
        st.info("Initiating Physical Math Execution Pipeline...")
        
        try:
            # Validate with Pydantic
            config = PulseConfig(
                target_smiles=target_smiles,
                run_mode=run_mode,
                output_dir=Path(os.environ.get("COCHEM_ARTIFACT_DIR", Path.home() / "cochem_artifacts"))
            )
            config.output_dir.mkdir(parents=True, exist_ok=True)
            
            # TODO: Integrate properly with pulse_core.dispatcher instead of subprocess
            st.warning("Pipeline execution logic needs integration with pulse_core.dispatcher.PulseDispatcher.")
            st.info("Awaiting proper orchestrator integration. Halted to prevent spoofing execution.")
            
        except Exception as e:
            st.error(f"Pipeline validation or setup crashed: {str(e)}")
            logger.error(f"Error: {e}")
            kill_zombie_processes()
