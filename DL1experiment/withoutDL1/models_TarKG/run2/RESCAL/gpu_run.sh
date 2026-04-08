#!/bin/bash
#SBATCH -A rnk@v100
#SBATCH --job-name=tar_e1r2nodl_RESCAL
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --output=tar_e1r2nodl_RESCAL_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t4

module load python
conda deactivate
conda activate torch_pyg

cd $WORK/dr_benchmark/DL1experiment/wihtoutDL1/models_TarKG/run2/RESCAL/

srun python $WORK/dr_benchmark/dev/run_training.py \
    --config params.yaml 
