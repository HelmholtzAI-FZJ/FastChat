#!/bin/bash

cd $HOME/FastChat
source sc_venv_template/activate.sh

python3 fastchat/serve/controller.py --host 0.0.0.0 --port 21001
