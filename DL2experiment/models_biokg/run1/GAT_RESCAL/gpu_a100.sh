#!/bin/bash
#SBATCH -A rnk@a100
#SBATCH --job-name=biokg_kgate_dl2_GAT_RESCAL
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=biokg_kgate_dl2_GAT_RESCAL_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-t3

module load arch/a100
module load python/3.12.7
source $WORK/KGATE/.venv/bin/activate

cd $WORK/dr_benchmark/DL2experiment/models_biokg/run1/GAT_RESCAL/

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python $WORK/dr_benchmark/dev/run_kgate.py \
    --config kgate_config.toml

