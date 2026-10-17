# Project 2 Instructions: Parallel Potential Energy Surface Scan via Slurm Job Arrays

## Objective
Run a 21-point symmetric $\text{O–H}$ bond stretch scan of $\text{H}_2\text{O}$ ($R_{\text{OH}} \in [0.70, 1.70]\text{ \AA}$) in parallel using a [Slurm job array](https://slurm.schedmd.com/sbatch.html) (`submit_pes_array.slurm`), `run_single_point.py`, and `collect_pes_results.sh`.

## Workflow Files
1. **`run_single_point.py`**: Maps `--task-id` ($0 \dots 20$) to $R_{\text{OH}} = 0.70 + 0.05 \times \text{task\_id}\text{ \AA}$ and writes `pes_point_<task_id>.csv`.
2. **`submit_pes_array.slurm`**: Dispatches `#SBATCH --array=0-20` with node-local `$TMPDIR` scratch.
3. **`collect_pes_results.sh`**: Merges and sorts all 21 points into `water_pes_scan.csv` and reports the equilibrium bond length.
