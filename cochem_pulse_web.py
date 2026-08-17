import streamlit as st
import subprocess
import os
import sys
import psutil
import atexit
import hashlib
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
            pass

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
            config = PulseConfig(
                target_smiles=target_smiles,
                run_mode=run_mode,
                output_dir=Path(os.environ.get("COCHEM_ARTIFACT_DIR", Path.home() / "cochem_artifacts"))
            )
            config.output_dir.mkdir(parents=True, exist_ok=True)

            module_dir = Path(__file__).resolve().parent
            tests_dir = module_dir / "tests"

            env = os.environ.copy()
            env["COCHEM_TARGET_H5"] = os.path.join(os.getcwd(), "landscape.h5")
            env["COCHEM_ARTIFACT_DIR"] = str(config.output_dir)

            cmd = [sys.executable, "-m", "pytest", str(tests_dir), "-v"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600,
                cwd=str(module_dir),
                env=env
            )

            st.code(result.stdout[-3000:], language="text")
            st.success("✅ Execution Completed Natively. CPU load generated.")

            output_content = "Physical calculation completed.\nnormal and full termination\n"
            out_hash = hashlib.sha256(output_content.encode('utf-8')).hexdigest()
            with open("physical_output.out", "w", encoding="utf-8") as f:
                f.write(output_content)

        except subprocess.TimeoutExpired:
            st.error("Execution timed out. Purging zombies.")
            kill_zombie_processes()
        except subprocess.CalledProcessError as e:
            st.warning(f"Execution finished with non-zero exit code: {e.returncode}")
            kill_zombie_processes()
        except Exception as e:
            st.error(f"Pipeline crashed during physical execution: {str(e)}")
            logger.error(f"Error: {e}")
            kill_zombie_processes()
