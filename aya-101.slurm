#!/bin/bash
#SBATCH --job-name=aya-101
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2

echo "I AM ON "$(hostname) " running Aya"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
# Avoid the ModuleNotFoundError

#source $BLABLADOR_DIR/sc_venv_falcon/activate.sh
source $BLABLADOR_DIR/sc_venv_2024/activate.sh

export PATH=$BLABLADOR_DIR:$PATH
export PYTHONPATH=$BLABLADOR_DIR:$PYTHONPATH
export NUMEXPR_MAX_THREADS=16

srun python3 $BLABLADOR_DIR/fastchat/serve/vllm_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port 31036 --worker-address http://$(hostname).fz-juelich.de:31036 \
     --num-gpus 2 \
     --host 0.0.0.0 \
     --model-path models/aya-101 \
     --model-names "alias-multilingual,aya-101" \
     # --load-8bit It's faster without it 
     #     --max-gpu-memory 22Gb \
