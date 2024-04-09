#!/bin/bash
#SBATCH --job-name=neural-chat-7
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1

echo "I AM ON "$(hostname) " running neural-chat-7"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
source $BLABLADOR_DIR/sc_venv_2024/activate.sh

export PATH=$BLABLADOR_DIR:$PATH
export PYTHONPATH=$BLABLADOR_DIR:$PYTHONPATH

#srun python3 fastchat/serve/model_worker.py \
python3 $BLABLADOR_DIR/fastchat/serve/vllm_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port 31033 --worker-address http://$(hostname).fz-juelich.de:31033 \
     --num-gpus 1 \
     --host 0.0.0.0 \
     --model-path models/neural-chat-7b-v3-1 \
     # --load-8bit It's faster without it 
     #     --max-gpu-memory 22Gb \
