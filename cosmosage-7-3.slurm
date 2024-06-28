#!/bin/bash
#SBATCH --job-name=cosmosage_v3_7
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# #SBATCH --time=100:00:00

echo "I AM ON "$(hostname) " running cosmosage 7 v3"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
# Avoid the ModuleNotFoundError

source $BLABLADOR_DIR/sc_venv_sglang2/activate.sh

export PATH=$BLABLADOR_DIR:$PATH
export PYTHONPATH=$BLABLADOR_DIR:$PYTHONPATH
export NUMEXPR_MAX_THREADS=16

#srun python3 fastchat/serve/model_worker.py \
srun python3 $BLABLADOR_DIR/fastchat/serve/model_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port 31038 --worker-address http://$(hostname).fz-juelich.de:31038 \
     --num-gpus 1 \
     --host 0.0.0.0 \
     --model-path models/cosmosage-v3 \
     --model-name "4 Cosmosage v3 - CosmoSage answers your cosmology questions (new version June 2024)"
     # --load-8bit It's faster without it 
     #     --max-gpu-memory 22Gb \
