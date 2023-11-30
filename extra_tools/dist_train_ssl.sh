#!/usr/bin/env bash

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS=4

CONFIG=projects/configs/unipad_abl/uvtr_convnext_s_vs0.1_c128_pretrain.py

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --no-validate

CONFIG=projects/configs/unipad_abl/uvtr_convnext_s_vs0.1_c128_finetune.py

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch
