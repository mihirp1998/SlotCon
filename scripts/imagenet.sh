#!/bin/bash

set -e
set -x

data_dir="/projects/katefgroup/datasets/ImageNet/"
output_dir="./output/imagenet_classify_73_2"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 12342 --nproc_per_node=4 \
    main_pretrain.py \
    --dataset ImageNet \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --arch resnet50_pretrained \
    --dim-hidden 4096 \
    --dim-out 256 \
    --num-prototypes 256 \
    --teacher-momentum 0.99 \
    --teacher-temp 0.07 \
    --group-loss-weight 0.5 \
    \
    --batch-size 128 \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 3 \
    --epochs 400 \
    --fp16 \
    \
    --print-freq 10 \
    --save-freq 2 \
    --auto-resume \
    --num-workers 8 \
    --seg-weight 0.0 \
    --class-weight 0.7 \
    --cont-weight 0.3  --do-only-classification
    # --d 
    # --resume output/slotcon_coco_r50_pretrained_s05c05_2_mask/current.pth --no-load-optim
    # --overfit --d --overfit --no-aug --min-scale 1.0