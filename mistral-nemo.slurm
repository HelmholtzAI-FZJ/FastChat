#!/bin/bash
#SBATCH --job-name=msnm2gpu
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2

#vllm 0.7.4 from source screwed things
export LD_LIBRARY_PATH=/p/haicluster/llama/FastChat/sc_venv_sglang2/venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH

echo "I AM ON "$(hostname) " running mistral - nemo "

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
# Avoid the ModuleNotFoundError

source $BLABLADOR_DIR/sc_venv_vllm8/activate.sh

export PATH=$BLABLADOR_DIR:$PATH
export PYTHONPATH=$BLABLADOR_DIR:$PYTHONPATH
export NUMEXPR_MAX_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_IGNORE_DISABLED_P2P=1

srun python3 $BLABLADOR_DIR/fastchat/serve/vllm_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port 31050 --worker-address http://$(hostname).fz-juelich.de:31050 \
     --num-gpus 2 \
     --host 0.0.0.0 \
     --model-path models/Mistral-Nemo-Instruct-2407 \
     --model-name "alias-fast-experimental,10 Mistral-Nemo-Instruct-2407 - Our fast-experimental - with a large context size" \
     --distributed-executor-backend=ray \
     --max-model-len 106430
     # --load-8bit It's faster without it 
     #     --max-gpu-memory 22Gb \
