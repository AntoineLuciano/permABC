#!/bin/bash
#SBATCH --job-name=OSUM_10
#SBATCH --output=../cluster/%j.OUT-%x.out
#SBATCH --error=../cluster/%j.ERROR-%x.out

#SBATCH --nodes=2
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=2
#SBATCH --distribution=block  # Meilleure répartition des tâches
#SBATCH --nodelist=clust9,clust10


K=10

echo "Host: $(hostname)"
echo "Running on $(date)"
echo "Tâches allouées : $SLURM_NTASKS"
echo "Cœurs par tâche : $SLURM_CPUS_PER_TASK"
echo "Nombre de nœuds : $SLURM_NNODES"
echo "Nombre de nœuds (job) : $SLURM_JOB_NUM_NODES"

for i in {0..5}
do
  echo "Lancement de la tâche $i"
  srun --exclusive -N1 -n1 /home/users/luciano/miniconda3/envs/permabc_env/bin/python perm_vs_osum.py $K $i &
done

wait
echo "All tasks completed."
echo "End of job on $(date)"
