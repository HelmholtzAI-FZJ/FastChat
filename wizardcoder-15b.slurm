#!/bin/bash
#SBATCH --job-name=WizardCoder-15B
#SBATCH --output=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --error=/p/haicluster/llama/FastChat/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1000:00:00
#SBATCH --gres=gpu:1

echo "I AM ON "$(hostname) " running WizardCoder-15B on 1 gpu"

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
source $BLABLADOR_DIR/sc_venv_template/activate.sh

#srun python3 $BLABLADOR_DIR/fastchat/serve/model_worker.py \
python3 $BLABLADOR_DIR/fastchat/serve/model_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port 31015 --worker http://$(hostname):31015 \
     --num-gpus 1 \
     --host 0.0.0.0 \
     --model-path models/wizardlm-WizardCoder-15B-V1.0 \
     --model-names "code-davinci-002,WizardCoder-15B-V1.0" \
     --load-8bit
