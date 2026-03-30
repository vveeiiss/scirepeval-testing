#! /bin/bash
#SBATCH --partition=gpu #use gpu partition
#SBATCH --gres=gpu:1 #the amount of gpu for your job
#SBATCH --ntasks=1 #number of parallel tasks
#SBATCH --mem=16G #the amount of CPU mem
#SBATCH --time=2-10 #[OPTIONAL], Default=1h. formatted running time. Avaliable specifiers: "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".



#SBATCH -o /home/mdavood/scirepeval-testing/outputs/%A_%a.out #where to write stdout
#SBATCH -e /home/mdavood/scirepeval-testing/err.e #where to write stderr

nvidia-smi

source ~/.bashrc
conda activate scip

# Ensure the env's C++ runtime is used (fixes GLIBCXX_3.4.29 not found).
SCIP_PREFIX="${CONDA_PREFIX:-/home/mdavood/anaconda3/envs/scip}"
export LD_LIBRARY_PATH="${SCIP_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

#######OPTIONAL#######
#Here you can specify env variables for your script
#export your python environment if you want to use locally installed libraries

#export PATH=/home/asoleim/anaconda3/bin:$PATH 
#source activate BART 
#export CUDA_HOME="/usr/local/cuda-9.2" 
#export PATH="${CUDA_HOME}/bin:${PATH}"
#export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH}"
#export LD_LIBRARY_PATH="/home/asoleim/cudalibs:/usr/lib64/nvidia:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

python /home/mdavood/scirepeval-testing/metrics.py