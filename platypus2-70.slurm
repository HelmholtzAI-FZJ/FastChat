#!/bin/bash
#SBATCH --job-name=Platypus2-70B-instruct
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:7

echo "I AM ON "$(hostname) " running Platypus2-70B-instruct"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
export LOGDIR=$BLABLADOR_DIR/logs
export NCCL_P2P_DISABLE=1 # 3090s do not support p2p

cd $BLABLADOR_DIR
source $BLABLADOR_DIR/sc_venv_template/activate.sh

srun python3 $BLABLADOR_DIR/fastchat/serve/model_worker.py \
     --controller http://haicluster1.fz-juelich.de:21001 \
     --port 31020 --worker http://$(hostname):31020 \
     --num-gpus 7 \
     --host 0.0.0.0 \
     --model-path /p/haicluster/llama/FastChat/models/Platypus2-70B-instruct \
     # --load-8bit
