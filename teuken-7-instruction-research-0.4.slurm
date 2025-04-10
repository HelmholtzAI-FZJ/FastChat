#!/bin/bash
#SBATCH --job-name=teuken704
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# #SBATCH --time=100:00:00

echo "I AM ON "$(hostname) " running Teuken-7B-instruct-research-v0.4"


export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
# Avoid the ModuleNotFoundError

source $BLABLADOR_DIR/sc_venv_vllm8/activate.sh

export PATH=$BLABLADOR_DIR:$PATH
export PYTHONPATH=$BLABLADOR_DIR:$PYTHONPATH
export NUMEXPR_MAX_THREADS=8

srun python3 $BLABLADOR_DIR/fastchat/serve/vllm_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port 31054 --worker-address http://$(hostname).fz-juelich.de:31054 \
     --num-gpus 1 \
     --host 0.0.0.0 \
     --model-path models/Teuken-7B-instruct-research-v0.4 \
     --model-name "alias-opengptx,1 - Teuken-7B-instruct-research-v0.4 - The OpenGPT-X model"
     # --load-8bit It's faster without it 
     #     --max-gpu-memory 22Gb \
