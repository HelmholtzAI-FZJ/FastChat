#!/bin/bash
#SBATCH --job-name=WizardCoder-15B
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1000:00:00
#SBATCH --gres=gpu:2

echo "I AM ON "$(hostname) " running WizardCoder-15B on 1 gpu"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
# Avoid the ModuleNotFoundError

source $BLABLADOR_DIR/sc_venv_2024/activate.sh

export PATH=$BLABLADOR_DIR:$PATH
export PYTHONPATH=$BLABLADOR_DIR:$PYTHONPATH
export NUMEXPR_MAX_THREADS=16


srun python3 $BLABLADOR_DIR/fastchat/serve/sglang_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port 31015 --worker http://$(hostname):31015 \
     --num-gpus 2 \
     --host 0.0.0.0 \
     --model-path models/wizardlm-WizardCoder-15B-V1.0 \
     --model-names "alias-code,WizardCoder-15B-V1.0" 
