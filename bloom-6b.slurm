#!/bin/bash
#SBATCH --job-name=bloom-6b4-clp-german-oasst-v0.1
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=720:00:00
#SBATCH --gres=gpu:1

echo "I AM ON "$(hostname) " running mpt bloom-6b4-clp-german-oasst-v0.1"

cd /p/haicluster/llama/FastChat
source sc_venv_template/activate.sh

srun python3 fastchat/serve/model_worker.py \
     --controller http://haicluster1.fz-juelich.de:21001 \
     --port 31021 --worker http://$(hostname):31021 \
     --num-gpus 1 \
     --host 0.0.0.0 \
     --model-path /p/haicluster/llama/FastChat/models/bloom-6b4-clp-german-oasst-v0.1

#     --max-gpu-memory 22Gb \
