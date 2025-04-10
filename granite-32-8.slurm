#!/bin/bash
#SBATCH --job-name=granite8
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# #SBATCH --time=100:00:00

echo "I AM ON "$(hostname) " running granite 8"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
# Avoid the ModuleNotFoundError

source $BLABLADOR_DIR/sc_venv_sglang2/activate.sh

export PATH=$BLABLADOR_DIR:$PATH
export PYTHONPATH=$BLABLADOR_DIR:$PYTHONPATH
export NUMEXPR_MAX_THREADS=8
export PORT=31061

srun python3 $BLABLADOR_DIR/fastchat/serve/vllm_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port $PORT --worker-address http://$(hostname).fz-juelich.de:$PORT \
     --num-gpus 1 \
     --host 0.0.0.0 \
     --model-path models/granite-3.2-8b-instruct-preview \
     --model-name "alias-ibm,10 - Granite - 8b - a new IBM model" \
     --max-model-len 36336 \
#     --config-format mistral --load-format mistral --tokenizer-mode mistral \
                # --load-8bit It's faster without it 
     #     --max-gpu-memory 22Gb \
