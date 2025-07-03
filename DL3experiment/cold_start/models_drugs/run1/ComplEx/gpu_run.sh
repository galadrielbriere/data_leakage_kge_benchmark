#!/bin/bash
#SBATCH -A rnk@v100
#SBATCH --job-name=zeroshot_r1_ComplEx
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --output=zeroshot_r1_ComplEx_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t4

module load python
conda deactivate
conda activate torch_pyg

cd $WORK/dr_benchmark/DL3experiment/cold_start/models_drugs/run1/ComplEx/

srun python $WORK/dr_benchmark/dev/run_training.py \
    --config params.yaml 
