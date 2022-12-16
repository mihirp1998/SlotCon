#!/bin/bash

set -e
set -x

data_dir="/home/mihirpd_google_com/datasets/coco"
output_dir="./output/slotcon_coco_r50_800ep"

CUDA_VISIBLE_DEVICES=0 torchrun --master_port 12348 --nproc_per_node=1 \
    main_pretrain.py \
    --dataset COCO \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --arch resnet50 \
    --dim-hidden 4096 \
    --dim-out 256 \
    --num-prototypes 100 \
    --teacher-momentum 0.99 \
    --teacher-temp 0.07 \
    --group-loss-weight 0.5 \
    \
    --batch-size 3 \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 800 \
    --fp16 \
    \
    --print-freq 10 \
    --save-freq 50 \
    --auto-resume \
    --num-workers 0
