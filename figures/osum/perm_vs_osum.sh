#!/bin/bash
# Usage: sbatch --job-name="OSUM_K$1" perm_vs_osum.sh K

#SBATCH --output=/home/users/luciano/script/permABC/figures/osum/cluster/%j.OUT-%x.out

#SBATCH --error=/home/users/luciano/script/permABC/figures/osum/cluster/%j.ERR-%x.out
#SBATCH --nodes=3
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=2

K=$1  # Récupère la valeur de K passée en argument
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"


echo "==== Job info ===="
echo "Job name      : $SLURM_JOB_NAME"
echo "Job ID        : $SLURM_JOB_ID"
echo "K             : $K"
echo "Host          : $(hostname)"
echo "Date          : $(date)"
echo "Tâches allouées     : $SLURM_NTASKS"
echo "Cœurs par tâche     : $SLURM_CPUS_PER_TASK"
echo "Nombre de nœuds     : $SLURM_NNODES"
echo "Nombre de nœuds job : $SLURM_JOB_NUM_NODES"
echo "==================="

# Lancement des 6 tâches en parallèle
for i in {0..5}; do
  echo "Lancement de la tâche $i"
  srun -N1 -n1 /home/users/luciano/miniconda3/envs/permabc_env/bin/python perm_vs_osum.py $K $i &
done

wait
echo "Toutes les tâches sont terminées."
echo "Fin du job : $(date)"
