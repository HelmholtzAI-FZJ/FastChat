#!/bin/bash
#SBATCH --job-name=WizardCoder-34B
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1000:00:00
#SBATCH --gres=gpu:4

echo "I AM ON "$(hostname) " running WizardCoder-Python-34B-V1.0"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
export LOGDIR=$BLABLADOR_DIR/logs
export NCCL_P2P_DISABLE=1 # 3090s do not support p2p

cd $BLABLADOR_DIR
source $BLABLADOR_DIR/sc_venv_vllm/activate.sh

srun python3 $BLABLADOR_DIR/fastchat/serve/vllm_worker.py \
     --controller http://haicluster1.fz-juelich.de:21001 \
     --port 31026 --worker-address http://$(hostname):31026 \
     --num-gpus 4 \
     --host 0.0.0.0 \
     --model-path models/WizardCoder-Python-34B-V1.0 \
     --model-names "code-davinci-002,WizardCoder-Python-34B-V1.0" \
     #--load-8bit
