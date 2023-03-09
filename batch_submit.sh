#!/bin/bash

# give a name to your job
#SBATCH -J "asr"

# set the working directory
#SBATCH --chdir=/home/vgdilu/speech_recognition

# print and save the output results
#SBATCH -o outputs/out_%a.txt

# print the errors in file below
#SBATCH -e outputs/err_%a.txt

# select the specific partition that you want to run the job on check with narvi the name can be something different than gpu
#SBATCH -p gpu

# number of cpu cores
#SBATCH -c 8

# time allocated for the job
#SBATCH -t 1-00:00:00

#number of gpus
#SBATCH --gres=gpu:1

#specific constarint for example for selecting a specific gpu type
#SBATCH --constraint=volta

# memory allocated to the job
#SBATCH --mem=32G

#optinal if you want to get an email when the job is done in that case you might need to give your email address
#SBATCH --mail-type=END

# the module needed to be loaded you at least need cuda and cudnn if you run on gpu
module purge
module load cuda

source ~/.bashrc
export PYTHONPATH=$PYTHONPATH:.
# this line below the name gpu should bbe changed to the name of conda environment that you build in your account on Narvi
conda activate privacy_preservation

# this runs your code so first make sure this batch file is in the same folder as main.py
python /home/vgdilu/speech_recognition/src/asr_main.py --index $SLURM_ARRAY_TASK_ID