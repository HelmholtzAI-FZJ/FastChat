#!/bin/bash
#SBATCH --job-name=code13
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:2

echo "I AM ON "$(hostname) " running codellama 13 with 1 gpus"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
export LOGDIR=$BLABLADOR_DIR/logs
export NCCL_P2P_DISABLE=1 # 3090s do not support p2p

cd $BLABLADOR_DIR
source $BLABLADOR_DIR/sc_venv_falcon/activate.sh


#srun python3 $BLABLADOR_DIR/fastchat/serve/vllm_worker.py \
srun python3 fastchat/serve/model_worker.py \
     --controller http://haicluster1.fz-juelich.de:21001 \
     --port 31025 --worker http://$(hostname):31025 \
     --num-gpus 2 \
     --host 0.0.0.0 \
     --model-path /p/haicluster/llama/FastChat/models/CodeLlama-13b-Instruct-hf \
     #--load-8bit


#     --max-gpu-memory 22Gb \
