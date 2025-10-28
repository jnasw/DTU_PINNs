#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J sweep_pinn
#BSUB -n 8
#BSUB -W 12:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err


echo "Running script..."
echo "Running script..."

nvidia-smi
module swap python3/3.10.2
module swap cuda/12.1

source ../../Thesis/venv/bin/activate

python create_dataset_d.py
python test_mul.py

