#!/bin/bash
#SBATCH --job-name=Plt2-vllm-70B-instruct
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:8

echo "I AM ON "$(hostname) " running Platypus2-70B-instruct on vllm"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
export LOGDIR=$BLABLADOR_DIR/logs
export NCCL_P2P_DISABLE=1 # 3090s do not support p2p

cd $BLABLADOR_DIR
source $BLABLADOR_DIR/sc_venv_vllm/activate.sh

#srun python3 $BLABLADOR_DIR/fastchat/serve/model_worker.py \
srun python3 $BLABLADOR_DIR/fastchat/serve/vllm_worker.py \
     --controller http://haicluster1.fz-juelich.de:21001 \
     --port 31024 --worker-address http://$(hostname):31024 \
     --num-gpus 8 \
     --host 0.0.0.0 \
     --model-path /p/haicluster/llama/FastChat/models/Platypus2-70B-instruct \
     # --load-8bit
