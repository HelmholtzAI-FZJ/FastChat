#!/bin/bash

export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
source $BLABLADOR_DIR/sc_venv_sglang2/activate.sh

export PATH=$BLABLADOR_DIR:$PATH
export PYTHONPATH=$BLABLADOR_DIR:$PYTHONPATH
export NUMEXPR_MAX_THREADS=8


python3 fastchat/serve/gradio_web_server_prompt.py \
        --model-list-mode=reload \
        --host 0.0.0.0 \
        --port 7860 \
        --controller-url http://helmholtz-blablador.fz-juelich.de:21001
