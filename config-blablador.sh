export BLABLADOR_DIR="/p/haicluster/llama/FastChat"
export LOGDIR=$BLABLADOR_DIR/logs
export NCCL_P2P_DISABLE=1 # 3090s do not support p2p
#export BLABLADOR_CONTROLLER=http://haicluster1.fz-juelich.de
# WHhile haicluster1 is down
export BLABLADOR_CONTROLLER=http://haicluster2.fz-juelich.de
export BLABLADOR_CONTROLLER_PORT=21001

