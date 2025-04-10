#!/bin/bash
#SBATCH --job-name=dcoder14
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
# #SBATCH --time=100:00:00

echo "I AM ON "$(hostname) "running DeepCoder-14B-Preview"

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
     --port 31053 --worker-address http://$(hostname).fz-juelich.de:31053 \
     --num-gpus 2 \
     --host 0.0.0.0 \
     --model-path models/DeepCoder-14B-Preview \
     --model-name "alias-code,3 - DeepCoder-14B-Preview - the code model from 09.04.2025" \
     --max-model-len 62928
     # --load-8bit It's faster without it 
     #     --max-gpu-memory 22Gb \
