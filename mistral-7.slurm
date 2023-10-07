#!/bin/bash
#SBATCH --job-name=mistral7
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1

echo "I AM ON "$(hostname) " running Mistral7"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
source $BLABLADOR_DIR/sc_venv_template/activate.sh

#srun python3 fastchat/serve/model_worker.py \
python3 fastchat/serve/model_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port 31029 --worker http://$(hostname):31029 \
     --num-gpus 1 \
     --host 0.0.0.0 \
     --model-path models/Mistral-7B-v0.1 \
     --model-names "copilot-codex,vicuna-7b-v.13" \
#     --max-gpu-memory 22Gb \
