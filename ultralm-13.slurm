#!/bin/bash
#SBATCH --job-name=UltraLM-13b
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1

echo "I AM ON "$(hostname) " running UltraLM-13b"

cd /p/haicluster/llama/FastChat
source sc_venv_template/activate.sh

# This model generates nonsense as of 2021-09-15

srun python3 -m fastchat.serve.model_worker \
     --controller http://haicluster1.fz-juelich.de:21001 \
     --port 31014 --worker http://$(hostname):31014 \
     --num-gpus 1 \
     --host 0.0.0.0 \
     --model-path /p/haicluster/llama/FastChat/models/UltraLM-13b \
     --load-8bit \ 


#     --max-gpu-memory 22Gb \
