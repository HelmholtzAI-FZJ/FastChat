#!/bin/bash
#SBATCH --job-name=GritLM-7B
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# #SBATCH --time=100:00:00

echo "I AM ON "$(hostname) " running GritLM-7B"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
# Avoid the ModuleNotFoundError

source $BLABLADOR_DIR/sc_venv_vllm8/activate.sh

export PATH=$BLABLADOR_DIR:$PATH
export PYTHONPATH=$BLABLADOR_DIR:$PYTHONPATH
export NUMEXPR_MAX_THREADS=16

#srun python3 fastchat/serve/model_worker.py \
srun python3 $BLABLADOR_DIR/fastchat/serve/model_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port 31011 --worker-address http://$(hostname).fz-juelich.de:31011 \
     --num-gpus 1 \
     --host 0.0.0.0 \
     --model-path models/GritLM-7B \
     --model-name "alias-embeddings,gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002,5 - GritLM-7B - For Chat AND Text Embeddings" \
     # --load-8bit It's faster without it 
     #     --max-gpu-memory 22Gb \
