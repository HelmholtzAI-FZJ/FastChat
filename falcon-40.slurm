#!/bin/bash
#SBATCH --job-name=falcon-40-instrudc
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:5

echo "I AM ON "$(hostname) " running falcon 40 instruct on 5 gpus"

echo "Falcon-40b is a pytorch2-only model"

cd /p/haicluster/llama/FastChat
source sc_venv_falcon/activate.sh

srun python3 fastchat/serve/model_worker.py \
     --controller http://haicluster1.fz-juelich.de:21001 \
     --port 31002 --worker http://$(hostname):31002 \
     --num-gpus 5 \
     --host 0.0.0.0 \
     --model-path models/falcon-40b-instruct \
     # --load-8bit # 8 bit is single-gpu only as of july 2023