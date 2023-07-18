#!/bin/bash
#SBATCH --job-name=vicuna33
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:4


echo "I AM ON "$(hostname) " running vicuna 33 with 4 gpus"

cd /p/haicluster/llama/FastChat
source sc_venv_falcon/activate.sh
module load NCCL # forces the latest NCCL after the pip env

export NCCL_P2P_DISABLE=1 # 3090s do not support p2p

export HF_HOME=/p/haicluster/llama/FastChat/models/

srun python3 fastchat/serve/vllm_worker.py \
     --controller http://haicluster1.fz-juelich.de:21001 \
     --port 31011 --worker-address http://$(hostname):31011 \
     --num-gpus 4 \
     --host 0.0.0.0 \
     --model-path models/vicuna-33b-v1.3 \
#     --max-gpu-memory 23Gb \
