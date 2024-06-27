#!/bin/bash

cd $HOME/FastChat
source sc_venv_template/activate.sh

python3 fastchat/serve/gradio_web_server_branded.py \
        --model-list-mode=reload \
        --host 0.0.0.0 \
        --port 7860 \
        --controller-url http://helmholtz-blablador.fz-juelich.de:21001
