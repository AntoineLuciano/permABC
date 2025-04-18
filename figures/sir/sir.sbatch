#!/bin/bash
#SBATCH --job-name=sir_job_3
#SBATCH --output=/home/users/luciano/script/permABC/figures/sir/cluster/%j.OUT-%x.out
#SBATCH --error=/home/users/luciano/script/permABC/figures/sir/cluster/%j.ERR-%x.out

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10


# Optimisation CPU pour JAX
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"
export XDG_CACHE_HOME="/tmp/$USER/cache"
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=.85

echo "Host : $(hostname)"
echo "Date : $(date)"
echo "Cœurs CPU alloués : ${SLURM_CPUS_PER_TASK}"

# Définir la seed
seed=2

# Lancement des scripts Python
/home/users/luciano/miniconda3/envs/permabc_env/bin/python /mnt/nfs/nrdata02-users-data/luciano/script/permABC/figures/sir/sir.py $seed
/home/users/luciano/miniconda3/envs/permabc_env/bin/python /mnt/nfs/nrdata02-users-data/luciano/script/permABC/figures/sir/sir_figures.py $seed

echo "Fin du job : $(date)"
