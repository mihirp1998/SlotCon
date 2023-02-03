#!/bin/bash

set -e
set -x

data_dir="/scratch/coco"
output_dir="./output/slotcon_coco_r50_pretrained_s05c05_2_mask"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 12342 --nproc_per_node=4 \
    main_pretrain.py \
    --dataset COCO \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --arch resnet50_maskformer \
    --dim-hidden 4096 \
    --dim-out 256 \
    --num-prototypes 256 \
    --teacher-momentum 0.99 \
    --teacher-temp 0.07 \
    --group-loss-weight 0.5 \
    \
    --batch-size 64 \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 800 \
    --fp16 \
    \
    --print-freq 10 \
    --save-freq 2 \
    --auto-resume \
    --num-workers 8 \
    --seg-weight 0.5 \
    --cont-weight 0.5 \
    # --resume output/slotcon_coco_r50_pretrained_s05c05_2/current.pth
    # --overfit 
    # --d
    
