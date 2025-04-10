#!/bin/bash
#SBATCH --job-name=trustllm1gpu
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

#vllm 0.7.4 from source screwed things
export LD_LIBRARY_PATH=/p/haicluster/llama/FastChat/sc_venv_sglang/venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH

echo "I AM ON "$(hostname) " running trustllm 7.8b"

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
     --port 31037 --worker-address http://$(hostname).fz-juelich.de:31037 \
     --num-gpus 1 \
     --host 0.0.0.0 \
     --model-path models/TrustLLM-7-8B-2-3T \
     --model-name "alias-trustllm,11 - TrustLLM preview 2" \
     --dtype bfloat16 \
     --max-model-len 4096 \
     --gpu-memory-utilization 0.8
     # --distributed-executor-backend=ray \
     # --load-8bit It's faster without it 
     #     --max-gpu-memory 22Gb \
