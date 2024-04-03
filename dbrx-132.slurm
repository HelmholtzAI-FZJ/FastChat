#!/bin/bash
#SBATCH --job-name=dbrx-instruct
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8

echo "I AM ON "$(hostname) " running dbrx-instruct"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
# Avoid the ModuleNotFoundError

source $BLABLADOR_DIR/sc_venv_sglang2/activate.sh
#source $BLABLADOR_DIR/sc_venv_2024/activate.sh

export PATH=$BLABLADOR_DIR:$PATH
export PYTHONPATH=$BLABLADOR_DIR:$PYTHONPATH

#srun python3 fastchat/serve/model_worker.py \
srun python3 $BLABLADOR_DIR/fastchat/serve/model_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port 31043 --worker-address http://$(hostname).fz-juelich.de:31043 \
     --num-gpus 8 \
     --host 0.0.0.0 \
     --model-path models/dbrx-instruct \
     --model-name "alias-large,dbrx-instruct" \
     # --load-8bit It's faster without it 
     #     --max-gpu-memory 22Gb \
