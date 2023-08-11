#!/bin/bash
#SBATCH --job-name=qcammel-13b
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:2

echo "I AM ON "$(hostname) " running qcammel-13b"

cd /p/haicluster/llama/FastChat
source sc_venv_template/activate.sh


srun python3 fastchat/serve/model_worker.py \
     --controller http://haicluster1.fz-juelich.de:21001 \
     --port 31023 --worker http://$(hostname):31023 \
     --num-gpus 2 \
     --host 0.0.0.0 \
     --model-path /p/haicluster/llama/FastChat/models/qCammel-13 \
     # --load-8bit
