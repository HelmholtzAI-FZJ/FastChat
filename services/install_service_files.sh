#!/bin/bash

sudo cp fastchat_controller.service /etc/systemd/system
sudo cp fastchat_api.service /etc/systemd/system
sudo cp fastchat_web.service /etc/systemd/system


sudo systemctl daemon-reload
sudo systemctl enable fastchat_controler.service
sudo systemctl enable fastchat_api.service
sudo systemctl enable fastchat_web.service
