#!/bin/bash
#SBATCH --job-name=Ll2-70bchat
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:7

echo "I AM ON "$(hostname) " running Llama-2-70b-chat-hf"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
export LOGDIR=$BLABLADOR_DIR/logs

cd $BLABLADOR_DIR
source $BLABLADOR_DIR/sc_venv_template/activate.sh

srun python3 $BLABLADOR_DIR/fastchat/serve/model_worker.py \
     --controller http://haicluster1.fz-juelich.de:21001 \
     --port 31019 --worker http://$(hostname):31019 \
     --num-gpus 7 \
     --host 0.0.0.0 \
     --model-path /p/haicluster/llama/FastChat/models/Llama-2-70b-chat-hf \
     # --load-8bit
