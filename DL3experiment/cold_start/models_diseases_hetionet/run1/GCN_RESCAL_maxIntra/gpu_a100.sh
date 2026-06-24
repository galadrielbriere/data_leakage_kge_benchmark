#!/bin/bash
#SBATCH -A rnk@a100
#SBATCH --job-name=hetionet_kgate_dl3dis_GCN_RESCAL_maxIntra
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=hetionet_kgate_dl3dis_GCN_RESCAL_maxIntra_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --hint=nomultithread
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-t3

module load arch/a100
module load python/3.12.7
source $WORK/KGATE/.venv/bin/activate

cd $WORK/dr_benchmark/DL3experiment/cold_start/models_diseases_hetionet/run1/GCN_RESCAL_maxIntra/
srun python $WORK/dr_benchmark/dev/run_kgate.py \
	--config params.toml

