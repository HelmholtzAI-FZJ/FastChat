#!/bin/bash
#SBATCH --job-name=LWM-Text-Chat-1M
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# #SBATCH --time=100:00:00

echo "I AM ON "$(hostname) " running LWM-Text-Chat-1M"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
# Avoid the ModuleNotFoundError

source $BLABLADOR_DIR/sc_venv_sglang2/activate.sh

export PATH=$BLABLADOR_DIR:$PATH
export PYTHONPATH=$BLABLADOR_DIR:$PYTHONPATH
export NUMEXPR_MAX_THREADS=8

#srun python3 fastchat/serve/model_worker.py \
srun python3 $BLABLADOR_DIR/fastchat/serve/sglang_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port 31044 --worker-address http://$(hostname).fz-juelich.de:31044 \
     --num-gpus 1 \
     --host 0.0.0.0 \
     --model-path models/LWM-Text-Chat-1M-Llama2 \
     --model-name "alias-long" \
#     --max-gpu-memory 20Gb \
#      --load-8bit # It's faster without it 
