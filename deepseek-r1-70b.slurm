#!/bin/bash
#SBATCH --job-name=deepskr18
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
# #SBATCH --time=100:00:00

echo "I AM ON "$(hostname) " running DeepSeek-R1-Distill-Llama-70B"

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
     --port 31059 --worker-address http://$(hostname).fz-juelich.de:31059 \
     --num-gpus 8 \
     --host 0.0.0.0 \
     --model-path models/DeepSeek-R1-Distill-Llama-70B \
     --model-name "alias-large,2 - DeepSeek-R1-Distill-Llama-70B - large model from january 2025" \
     --gpu-memory-utilization 0.85 \
     --max-model-len 28560 \
     # --conv-template deepseek-chat
     # --cpu-offload-gb 10
     # --load-8bit It's faster without it 

