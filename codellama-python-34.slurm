#!/bin/bash
#SBATCH --job-name=code13
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:4

echo "I AM ON "$(hostname) " running codellama 34 with 4 gpus"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
source $BLABLADOR_DIR/sc_venv_falcon/activate.sh

#srun python3 $BLABLADOR_DIR/fastchat/serve/vllm_worker.py \
#srun python3 fastchat/serve/model_worker.py \
python3 fastchat/serve/model_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port 31025 --worker http://$(hostname):31025 \
     --num-gpus 4 \
     --host 0.0.0.0 \
     --model-path /p/haicluster/llama/FastChat/models/CodeLlama-34b-Python-hf \
     #--load-8bit


#     --max-gpu-memory 22Gb \
