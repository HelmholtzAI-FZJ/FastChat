#!/bin/bash
#SBATCH --job-name=codefuse-34
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:4

echo "I AM ON "$(hostname) " running CodeFuse-CodeLlama-34B in 4 gpus"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
export LOGDIR=$BLABLADOR_DIR/logs
export NCCL_P2P_DISABLE=1 # 3090s do not support p2p

cd $BLABLADOR_DIR
source $BLABLADOR_DIR/sc_venv_falcon/activate.sh


#srun python3 $BLABLADOR_DIR/fastchat/serve/vllm_worker.py \
srun python3 fastchat/serve/model_worker.py \
     --controller http://haicluster1.fz-juelich.de:21001 \
     --port 31026 --worker http://$(hostname):31026 \
     --num-gpus 4 \
     --host 0.0.0.0 \
     --model-path /p/haicluster/llama/FastChat/models/CodeFuse-CodeLlama-34B \
     #--load-8bit


#     --max-gpu-memory 22Gb \
