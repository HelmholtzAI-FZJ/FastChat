#!/bin/bash
#SBATCH --job-name=longchat13
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1

echo "I AM ON "$(hostname) " running longchat-13b-16k"

cd /p/haicluster/llama/FastChat
source sc_venv_template/activate.sh


# THIS MODEL DOES NOT WORK PROPERLY YET

srun python3 fastchat/serve/model_worker.py \
     --controller http://haicluster1.fz-juelich.de:21001 \
     --port 31006 --worker http://$(hostname):31006 \
     --num-gpus 1 \
     --host 0.0.0.0 \
     --model-path /p/haicluster/llama/FastChat/models/longchat-13b-16k \
     --load-8bit
     # --max-gpu-memory 22Gb \
