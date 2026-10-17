# Module 02: SSH, HPC Clusters & Slurm Job Scheduling

Our group runs large-scale electronic structure and machine learning workloads on Linux clusters managed by [Slurm Workload Manager](https://www.google.com/search?q=https%3A%2F%2Fslurm.schedmd.com%2Fquickstart.html), including the [Mississippi State University HPC² clusters](https://www.google.com/search?q=https%3A%2F%2Fwww.hpc.msstate.edu%2Fcomputing%2Fhpc.php) ([Orion and Hercules](https://www.google.com/search?q=https%3A%2F%2Fdocs.rdhpcs.noaa.gov%2Fsystems%2FMSU-HPC_user_guide.html) and [Atlas and Morrill](https://www.google.com/search?q=https%3A%2F%2Fdocs.hpc.msstate.edu%2Fcluster%2Findex.html)).

## 1. Setting Up SSH Keys & `~/.ssh/config`

1. **Generate an Ed25519 SSH Key Pair** on your laptop ([NOAA/MSU SSH Guide](https://www.google.com/search?q=https%3A%2F%2Fdocs.rdhpcs.noaa.gov%2Fconnecting%2Findex.html)):
   ```bash
   ssh-keygen -t ed25519 -C "your_netid@msstate.edu"
   ```
2. **Create `~/.ssh/config`** so you can log in with a short alias (`ssh cluster`) and reuse multiplexed connections:
   ```sshconfig
   Host cluster
       HostName login.hpc.msstate.edu
       User your_netid
       IdentityFile ~/.ssh/id_ed25519
       ControlMaster auto
       ControlPath ~/.ssh/cm-%r@%h:%p
       ControlPersist 4h
       ServerAliveInterval 60
   ```
3. **Transferring Files Between Laptop and Cluster**:
   * Use `rsync -avzP ./project_folder/ cluster:/work/your_group/your_netid/project_folder/` to sync only changed files.
   * Never transfer multi-gigabyte `.chk` or `.npy` two-electron integral scratch files over Wi-Fi; transfer compact `.csv`, `.json`, or `.h5` summaries instead.

## 2. Cluster Etiquette, Storage Tiers & `Lmod` Modules

* **Rule #1: Never Run Calculations on the Login Node.** Login nodes are strictly for editing scripts, managing Git repositories, and submitting jobs. Any calculation lasting more than a few seconds must go through [sbatch](https://www.google.com/search?q=https%3A%2F%2Fslurm.schedmd.com%2Fsbatch.html) or an interactive compute allocation (`salloc --nodes=1 --cpus-per-task=4 --time=01:00:00`).
* **Three Storage Tiers**:
  * **Home (`/home/$USER`)**: Small quota, backed up. Keep source code, dotfiles, and small scripts here.
  * **Work (`/work/...`)**: Parallel Lustre storage ([MSU-HPC User Guide](https://www.google.com/search?q=https%3A%2F%2Fdocs.rdhpcs.noaa.gov%2Fsystems%2FMSU-HPC_user_guide.html)) for active datasets and calculation outputs.
  * **Node-Local Scratch (`$TMPDIR` or `/scratch`)**: Fast local SSD on the compute node. Always set `export PYSCF_TMPDIR=$TMPDIR` inside your Slurm script so 4-index integral I/O stays off the shared network filesystem.
* **Essential Slurm Commands**:
  * [sbatch submit.slurm](https://www.google.com/search?q=https%3A%2F%2Fslurm.schedmd.com%2Fsbatch.html): Submit a batch script.
  * [squeue -u \$USER](https://www.google.com/search?q=https%3A%2F%2Fslurm.schedmd.com%2Fsqueue.html): Check running (`R`) and pending (`PD`) jobs.
  * `scancel <jobid>`: Cancel a job (or `scancel -u $USER` to cancel all your jobs).
  * `seff <jobid>`: Inspect CPU and RAM efficiency after a job finishes so you don't over-request memory.

## 3. Example Project 2: Parallel Potential Energy Surface Scan (`submit_pes_array.slurm`)

Instead of running 21 bond lengths sequentially in a single job, inspect and submit `submit_pes_array.slurm`. It uses `#SBATCH --array=0-20` so Slurm dispatches 21 independent single-point [PySCF](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2Fuser%2Findex.html) tasks simultaneously, each reading its bond length index from `$SLURM_ARRAY_TASK_ID` and writing to node-local scratch `$TMPDIR` before copying back the converged energy.
