#!/bin/bash
#SBATCH --job-name=ll33nm48
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
# #SBATCH --time=100:00:00

#vllm 0.7.4 from source screwed things
export LD_LIBRARY_PATH=/p/haicluster/llama/FastChat/sc_venv_sglang2/venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH


echo "I AM ON "$(hostname) " running Llama-3_3-Nemotron-Super-49B-v1"
export SQLITE_TMPDIR=/tmp
export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
# Avoid the ModuleNotFoundError

source $BLABLADOR_DIR/sc_venv_sglang/activate.sh

export PATH=$BLABLADOR_DIR:$PATH
export PYTHONPATH=$BLABLADOR_DIR:$PYTHONPATH
export NUMEXPR_MAX_THREADS=8

srun python3 $BLABLADOR_DIR/fastchat/serve/vllm_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port 31069 --worker-address http://$(hostname).fz-juelich.de:31069 \
     --num-gpus 8 \
     --host 0.0.0.0 \
     --model-path models/Llama-3_3-Nemotron-Super-49B-v1 \
     --model-name "alias-large,2 - Llama-3_3-Nemotron-Super-49B-v1 - an improved version of llama3.3-70" \
#     --max-model-len 15768
     # --load-8bit It's faster without it 
     #     --max-gpu-memory 22Gb \
