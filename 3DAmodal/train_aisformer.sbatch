#!/usr/bin/env bash
#SBATCH --mem  120GB
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 16
#SBATCH --time 48:00:00
#SBATCH --constrain "arwen"
#SBATCH --mail-type BEGIN,FAIL,END
#SBATCH --mail-user tibbe@kth.se
#SBATCH --output /Midgard/home/%u/3damodal/logs/%J_slurm.out
#SBATCH --error  /Midgard/home/%u/3damodal/logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. /Midgard/home/tibbe/mambaforge/etc/profile.d/conda.sh
conda activate aisformer
python /Midgard/home/tibbe/3damodal/3DAmodal/train_aisformer.py --config /Midgard/home/tibbe/3damodal/3DAmodal/configs/config.yaml