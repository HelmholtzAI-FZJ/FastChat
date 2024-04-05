#!/bin/bash

cd $FAST

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR

source $BLABLADOR_DIR/sc_venv_2024/activate.sh

# Avoid the ModuleNotFoundError
export PATH=$BLABLADOR_DIR:$PATH
export PYTHONPATH=$BLABLADOR_DIR:$PYTHONPATH


python3 fastchat/serve/gradio_web_server_branded.py \
        --model-list-mode=reload \
        --host 0.0.0.0 \
        --port 7860 \
        --controller-url ${BLABLADOR_CONTROLLER}:${BLABLADOR_CONTROLLER_PORT}

