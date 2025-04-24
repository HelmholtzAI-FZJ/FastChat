#!/bin/bash

cd $HOME/FastChat
source sc_venv_template/activate.sh

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
# Avoid the ModuleNotFoundError

export PATH=$BLABLADOR_DIR:$PATH
export PYTHONPATH=$BLABLADOR_DIR:$PYTHONPATH

#_branded.py \
python3 fastchat/serve/gradio_web_server.py \
        --model-list-mode=reload \
        --host 0.0.0.0 \
        --port 7860 \
        --controller-url http://helmholtz-blablador.fz-juelich.de:21001 \
        --vision-arena
