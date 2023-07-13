#!/bin/bash
#SBATCH --job-name=vicuna7
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1

echo "I AM ON "$(hostname) " running vicuna 7"

cd /p/haicluster/llama/FastChat
source sc_venv_template/activate.sh

# We use model-names to fool langchain, which expects model names from OpenAI
# This messes up with the web server, as seen in https://github.com/lm-sys/FastChat/issues/1947
srun python3 -m fastchat.serve.model_worker \
     --controller http://haicluster1.fz-juelich.de:21001 \
     --port 31012 --worker http://$(hostname):31012 \
     --num-gpus 1 \
     --host 0.0.0.0 \
     --model-path models/vicuna-7b-v1.3 \
     # --model-names "gpt-3.5-turbo,vicuna-7b-v.13" \
     

#     --max-gpu-memory 22Gb \
