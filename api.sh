#/bin/bash

cd /p/haicluster/llama/FastChat
source sc_venv_template/activate.sh

python3 fastchat/serve/openai_api_server.py --host 0.0.0.0 --port 8000

