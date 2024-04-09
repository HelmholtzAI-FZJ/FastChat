#!/bin/bash
#SBATCH --job-name=mistral7
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
# #SBATCH --time=100:00:00
#SBATCH --gres=gpu:1

##### Parameters for a model
# nodes? 
# gpus
# Path
# Alias
# port

echo "I AM ON "$(hostname) " running $1"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
# Avoid the ModuleNotFoundError

#source $BLABLADOR_DIR/sc_venv_falcon/activate.sh
source $BLABLADOR_DIR/sc_venv_2024/activate.sh

export PATH=$BLABLADOR_DIR:$PATH
export PYTHONPATH=$BLABLADOR_DIR:$PYTHONPATH
export NUMEXPR_MAX_THREADS=16

srun python3 $BLABLADOR_DIR/fastchat/serve/sglang_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port 31029 --worker-address http://$(hostname).fz-juelich.de:31029 \
     --num-gpus 1 \
     --host 0.0.0.0 \
     --model-path models/Mistral-7B-Instruct-v0.2 \
     --model-names "alias-fast,Mistral-7B-Instruct-v0.2" \
     # --load-8bit It's faster without it 
     #     --max-gpu-memory 22Gb \
