#!/bin/bash

cd $HOME/FastChat
source sc_venv_template/activate.sh

python3 fastchat/serve/openai_api_server.py --host 0.0.0.0 --port 18000 --controller-address http://localhost:21001
