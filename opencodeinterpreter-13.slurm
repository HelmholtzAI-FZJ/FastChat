#!/bin/bash
#SBATCH --job-name=opencodeinterpreter13
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
# #SBATCH --time=100:00:00

echo "I AM ON "$(hostname) " running opencodeinterpreter 7"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
# Avoid the ModuleNotFoundError

source $BLABLADOR_DIR/sc_venv_sglang/activate.sh

export PATH=$BLABLADOR_DIR:$PATH
export PYTHONPATH=$BLABLADOR_DIR:$PYTHONPATH
export NUMEXPR_MAX_THREADS=16

#srun python3 fastchat/serve/model_worker.py \
srun python3 $BLABLADOR_DIR/fastchat/serve/vllm_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port 31029 --worker-address http://$(hostname).fz-juelich.de:31029 \
     --num-gpus 2 \
     --host 0.0.0.0 \
     --model-path models/OpenCodeInterpreter-CL-13B \
     # --load-8bit It's faster without it 
     #     --max-gpu-memory 22Gb \
