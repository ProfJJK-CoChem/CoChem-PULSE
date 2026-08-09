import json

def dispatch_swarm_to_node(wigner_seeds: tuple):
    """
    Formats 100+ separate execution payloads and hands them to NODE with the [CPU_DIST] tag.
    Reinstates AIMNet2-NSE for Non-Adiabatic Surface Hopping (NASH) molecular dynamics,
    fully removing legacy sTDA/FOMO-CI hacks.
    """
    coords, momenta = wigner_seeds
    jobs = []
    
    for i in range(len(coords)):
        job = {
            "job_id": f"traj_{i}",
            "coordinates": coords[i].tolist(),
            "momenta": momenta[i].tolist(),
            "engine": "AIMNet2-NSE",  # Strict AIMNet2-NSE for NASH
            "tags": ["[CPU_DIST]", "[NASH]"]
        }
        jobs.append(job)
        
    print(f"Dispatched {len(jobs)} NASH AIMNet2-NSE trajectories to swarm.")
    return jobs
