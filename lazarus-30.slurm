#!/bin/bash
#SBATCH --job-name=30B-Lazarus
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:4


echo "I AM ON "$(hostname) " running 30B-Lazarus with 4 gpus"

cd /p/haicluster/llama/FastChat
source sc_venv_template/activate.sh


srun python3 fastchat/serve/model_worker.py \
     --controller http://haicluster1.fz-juelich.de:21001 \
     --port 31004 --worker http://$(hostname):31004 \
     --num-gpus 4 \
     --host 0.0.0.0 \
     --model-path /p/haicluster/llama/FastChat/models/30B-Lazarus \
     # --load-8bit # No benefit, still needs 4 gpus on 18/07/2023
     # --max-gpu-memory 22Gb \
