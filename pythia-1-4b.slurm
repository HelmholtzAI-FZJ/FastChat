#!/bin/bash
#SBATCH --job-name=pythia-1.4b-deduped
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1

echo "I AM ON "$(hostname) " running mpt pythia-1.4b-deduped"

cd /p/haicluster/llama/FastChat
source sc_venv_template/activate.sh


srun python3 fastchat/serve/model_worker.py \
     --controller http://haicluster1.fz-juelich.de:21001 \
     --port 31009 --worker http://$(hostname):31009 \
     --num-gpus 1 \
     --host 0.0.0.0 \
     --model-path /p/haicluster/llama/FastChat/models/pythia-1.4b-deduped

#     --max-gpu-memory 22Gb \
