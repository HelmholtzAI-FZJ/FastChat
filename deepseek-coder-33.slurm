#!/bin/bash
#SBATCH --job-name=deepseek-coder-33
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:4

echo "I AM ON "$(hostname) " running deepseek-coder-33"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
export PATH=$PATH:$BLABLADOR_DIR
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
source $BLABLADOR_DIR/sc_venv_falcon/activate.sh

#srun python3 fastchat/serve/model_worker.py \
#python3 $BLABLADOR_DIR/fastchat/serve/vllm_worker.py \
python3 fastchat/serve/model_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port 31034 --worker-address http://$(hostname).fz-juelich.de:31034 \
     --num-gpus 4 \
     --host 0.0.0.0 \
     --model-path models/deepseek-coder-33b-instruct \
     # --load-8bit It's faster without it 
     #     --max-gpu-memory 22Gb \
