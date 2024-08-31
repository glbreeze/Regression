#!/bin/bash

#SBATCH --job-name=lr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000


# job info

NOISE=$1
WD=$2
LOSS=$3



# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/lg154/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
--overlay /scratch/lg154/sseg/dataset/tiny-imagenet-200.sqf:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
python main.py --dataset reacher --data_ratio 0.1 --arch mlp256_256_256 --which_y -1\
    --act relu --bn p --loss l2 --noise_ratio ${NOISE} \
    --ufm --feat f --wd ${WD} --lambda_H ${WD} --lambda_W ${WD} \
    --max_epoch 600 --batch_size 512 --lr 2e-3 --warmup 0 \
    --seed 2021 --log_freq 10 --save_freq -1\
 --exp_name re0.1_ufm_noise${NOISE}_wd${WD}
 " 


 # --bias 
 # --act ${ACT}