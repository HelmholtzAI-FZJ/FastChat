#/bin/bash

cd /p/haicluster/llama/FastChat
source sc_venv_template/activate.sh

#python3 -m fastchat.serve.gradio_web_server_multi --share
#python3 -m fastchat.serve.gradio_web_server --model-list-mode=reload --host 0.0.0.0
export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE=http://localhost:8000/v1

python3 fastchat/serve/gradio_web_server_branded.py \
        --share \
        --model-list-mode=reload \
        --host 0.0.0.0 \
